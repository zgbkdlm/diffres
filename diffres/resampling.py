import jax
import jax.numpy as jnp
from diffres.integators import lord_and_rougemont, jentzen_and_kloeden, tweedie
from diffres.typings import JArray, JKey
from functools import partial


def _sorted_uniforms(n, key: JKey) -> JArray:
    # Credit goes to Nicolas Chopin
    us = jax.random.uniform(key, (n + 1,))
    z = jnp.cumsum(-jnp.log(us))
    return z[:-1] / z[-1]


def _systematic_or_stratified(key: JKey, weights: JArray, is_systematic: bool) -> JArray:
    n = weights.shape[0]
    if is_systematic:
        u = jax.random.uniform(key, ())
    else:
        u = jax.random.uniform(key, (n,))
    idx = jnp.searchsorted(jnp.cumsum(weights),
                           (jnp.arange(n, dtype=weights.dtype) + u) / n)
    return jnp.clip(idx, 0, n - 1)


def systematic(key: JKey, weights: JArray) -> JArray:
    return _systematic_or_stratified(weights, key, True)


def stratified(key: JKey, weights: JArray) -> JArray:
    return _systematic_or_stratified(key, weights, False)


def multinomial(key: JKey, weights: JArray) -> JArray:
    """Not tested.
    """
    n = weights.shape[0]
    idx = jnp.searchsorted(jnp.cumsum(weights),
                           _sorted_uniforms(n, key))
    return jnp.clip(idx, 0, n - 1)


def diffusion_resampling(key: JKey, log_ws: JArray, samples: JArray, a: float, ts: JArray,
                         integrator: str = 'euler'):
    """Differentiable resampling using ensemble score.

    Parameters
    ----------
    key : JKey
        A JAX random key.
    log_ws : JArray (n, )
        Weights.
    samples : JArray (n, ...)
        Particles.
    a : float
        The forward noising parameter, must be negative.
    ts : JArray (nsteps + 1, )
        Time steps t0, t1, ..., tnsteps.
    integrator : str
        The SDE integrator.
    """
    n = log_ws.shape[0]
    data_shape = samples.shape[1:]
    nsteps = ts.shape[0] - 1
    ws = jnp.exp(log_ws)
    mu = jnp.einsum('i,i...->...', ws, samples)
    stat_vars = jnp.einsum('i,i...->...', ws, (samples - mu) ** 2)
    b2 = -stat_vars / (2 * a)
    T = ts[-1]

    def fwd_coeffs(x0, t):
        """
        x0 : (n, ...)
        """
        semigroup = jnp.exp(a * t)
        mt = x0 * semigroup + mu * (1 - semigroup)
        sig2t = stat_vars * (1 - semigroup ** 2)
        return mt, sig2t

    def logpdf_trans(x, mts, sig2ts):
        """(...,), (n, ...), (n, ...) -> (n, )"""
        return jnp.sum(jax.scipy.stats.norm.logpdf(x, mts, sig2ts ** 0.5).reshape(n, -1), axis=-1)

    def s(x, t):
        """
        (..., ), () -> (..., )
        """
        mts, sig2ts = fwd_coeffs(samples, t)  # (n, ...), (n, ...)
        log_alps = log_ws + logpdf_trans(x, mts, sig2ts)  # (n, )
        log_alps = log_alps - jax.scipy.special.logsumexp(log_alps)
        return jnp.einsum('i,i...->...', jnp.exp(log_alps), -(x - mts) / sig2ts)

    # Euler--Maruyama
    key, _ = jax.random.split(key)
    xTs = mu + stat_vars ** 0.5 * jax.random.normal(key, (n, *data_shape))

    def scan_body(carry, elem):
        xk = carry
        t_km1, tk, key_k = elem

        dt = tk - t_km1
        rnd = jax.random.normal(key_k, (n, *data_shape))
        if integrator == 'euler':
            x_km1 = xk + (-a * (xk - mu) + b2 * jax.vmap(s, in_axes=[0, None])(xk, tk)) * dt + (b2 * dt) ** 0.5 * rnd
        elif integrator == 'lord_and_rougemont':
            drift = lambda x, t: a * mu + b2 * jax.vmap(s, in_axes=[0, None])(x, T - tk)
            m, scale = lord_and_rougemont(-a, drift, b2 ** 0.5, xk, tk, dt)
            x_km1 = m + scale * rnd
        elif integrator == 'jentzen_and_kloeden':
            drift = lambda x, t: -a * mu + b2 * jax.vmap(s, in_axes=[0, None])(x, tk)
            m, scale = jentzen_and_kloeden(a, drift, b2 ** 0.5, xk, tk, dt)
            x_km1 = m + scale * rnd
        elif integrator == 'tweedie':
            sg = jnp.exp(a * dt)
            trans_var = stat_vars * (1 - sg ** 2)
            x_km1 = tweedie(sg, trans_var, s, mu, xk, tk) + (b2 * dt) ** 0.5 * rnd
        else:
            raise ValueError(f'Unknown integrator {integrator}.')
        return x_km1, None
    key, _ = jax.random.split(key)
    keys = jax.random.split(key, num=nsteps)
    x0s, _ = jax.lax.scan(scan_body, xTs, (ts[:-1], ts[1:], keys), reverse=True)
    return x0s

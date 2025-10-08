import jax
import jax.numpy as jnp
from diffres.integators import euler_maruyama, lord_and_rougemont, jentzen_and_kloeden, tweedie
from diffres.typings import JArray, JKey
from typing import Tuple


def _sorted_uniforms(n, key: JKey) -> JArray:
    # Credit goes to Nicolas Chopin
    us = jax.random.uniform(key, (n + 1,))
    z = jnp.cumsum(-jnp.log(us))
    return z[:-1] / z[-1]


def _systematic_or_stratified(key: JKey, weights: JArray, samples: JArray,
                              is_systematic: bool) -> Tuple[JArray, JArray]:
    n = weights.shape[0]
    if is_systematic:
        u = jax.random.uniform(key, ())
    else:
        u = jax.random.uniform(key, (n,))
    idx = jnp.searchsorted(jnp.cumsum(weights),
                           (jnp.arange(n, dtype=weights.dtype) + u) / n)
    inds = jnp.clip(idx, 0, n - 1)
    return jnp.full((n,), -jnp.log(n)), samples[inds]


def systematic(key: JKey, log_weights: JArray, samples: JArray) -> Tuple[JArray, JArray]:
    return _systematic_or_stratified(key, jnp.exp(log_weights), samples, True)


def stratified(key: JKey, log_weights: JArray, samples: JArray) -> Tuple[JArray, JArray]:
    return _systematic_or_stratified(key, jnp.exp(log_weights), samples, False)


def multinomial(key: JKey, log_weights: JArray, samples: JArray) -> Tuple[JArray, JArray]:
    weights = jnp.exp(log_weights)
    n = weights.shape[0]
    idx = jnp.searchsorted(jnp.cumsum(weights),
                           _sorted_uniforms(n, key))
    inds = jnp.clip(idx, 0, n - 1)
    return jnp.full((n,), -jnp.log(n)), samples[inds]


def multinomial_stopped():
    pass


def diffusion_resampling(key: JKey, log_ws: JArray, samples: JArray, a: float, ts: JArray,
                         integrator: str = 'lord_and_rougemont') -> Tuple[JArray, JArray]:
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

    #TODO: Double-check the routine for datashape
    #TODO: Make efficient parallel implementation
    #TODO: Efficient grad propagation
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

    def f(x, t):
        return a * mu + b2 * jax.vmap(s, in_axes=[0, None])(x, T - t)

    def drift(x, t):
        return -a * x + f(x, t)

    # SDE simulation
    key, _ = jax.random.split(key)
    xTs = mu + stat_vars ** 0.5 * jax.random.normal(key, (n, *data_shape))

    def scan_body(carry, elem):
        x = carry
        t_km1, tk, key_k = elem

        dt = tk - t_km1
        rnd = jax.random.normal(key_k, (n, *data_shape))
        if integrator == 'euler':
            m, scale = euler_maruyama(drift, b2 ** 0.5, x, t_km1, dt)
        elif integrator == 'lord_and_rougemont':
            m, scale = lord_and_rougemont(-a, f, b2 ** 0.5, x, t_km1, dt)
        elif integrator == 'jentzen_and_kloeden':
            m, scale = jentzen_and_kloeden(-a, f, b2 ** 0.5, x, t_km1, dt)
        else:
            raise ValueError(f'Unknown integrator {integrator}.')
        return m + scale * rnd, None

    key, _ = jax.random.split(key)
    keys = jax.random.split(key, num=nsteps)
    x0s, _ = jax.lax.scan(scan_body, xTs, (ts[:-1], ts[1:], keys))
    return jnp.full((n,), -jnp.log(n)), x0s

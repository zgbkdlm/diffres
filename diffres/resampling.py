import jax
import jax.numpy as jnp
from diffres.integators import euler_maruyama, lord_and_rougemont, jentzen_and_kloeden, tweedie
from diffres.typings import JArray, JKey
from diffrax import diffeqsolve, ODETerm, Dopri5, Euler
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
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


def _multinomial(key: JKey, log_weights: JArray) -> JArray:
    # weights = jnp.exp(log_weights)
    # n = weights.shape[0]
    # idx = jnp.searchsorted(jnp.cumsum(weights),
    #                        _sorted_uniforms(n, key))
    # return jnp.clip(idx, 0, n - 1)
    n = log_weights.shape[0]
    return jax.random.choice(key, n, shape=(n,), replace=True, p=jnp.exp(log_weights))


def multinomial(key: JKey, log_weights: JArray, samples: JArray) -> Tuple[JArray, JArray]:
    n = log_weights.shape[0]
    inds = _multinomial(key, log_weights)
    return jnp.full((n,), -jnp.log(n)), samples[inds]


def multinomial_stopped(key: JKey, log_weights: JArray, samples: JArray) -> Tuple[JArray, JArray]:
    n = log_weights.shape[0]
    inds = _multinomial(key, jax.lax.stop_gradient(log_weights))
    return jnp.full((n,), -jnp.log(n)) + log_weights[inds] - jax.lax.stop_gradient(log_weights[inds]), samples[inds]


def ensemble_ot(key: JKey, log_ws: JArray, samples: JArray, eps: float = None) -> Tuple[JArray, JArray]:
    """Entropic OT resampling.

    Parameters
    ----------
    key : JKey
        A JAX random key.
    log_ws : JArray (n, )
        Log weights.
    samples : JArray (n, d)
        Particles.
    eps : float
        Entropic regularization parameter. If None, set to 1 / log(n).

    Returns
    -------
    (n, ), (n, d)
        New log weights and particles.

    References
    ----------
    Adrien Corenflos, James Thornton, George Deligiannidis, Arnaud Doucet.
    Proceedings of the 38th International Conference on Machine Learning, PMLR 139:2100-2111, 2021.
    """
    n = log_ws.shape[0]
    if eps is None:
        eps = 1 / jnp.log(n)
    geom = pointcloud.PointCloud(samples, samples, epsilon=eps)
    prob = linear_problem.LinearProblem(geom,
                                        a=jnp.full((n,), 1 / n), b=jnp.exp(log_ws))
    solver = sinkhorn.Sinkhorn()
    out = solver(prob)
    return jnp.full((n,), -jnp.log(n)), out.matrix @ samples * n


def soft_resampling(key: JKey, log_ws, samples: JArray, alpha: float) -> Tuple[JArray, JArray]:
    """Soft resampling.

    Parameters
    ----------
    key : JKey
        A JAX random key.
    log_ws : JArray (n, )
        Log weights.
    samples : JArray (n, ...)
        Particles.
    alpha : float
        The softening parameter, must be in [0, 1]. alpha = 1 means no softening.

    Returns
    -------
    (n, ), (n, d)
        New log weights and particles.

    References
    ----------
    Karkus, P., Hsu, D., & Lee, W. S. (2018).
    Particle filter networks with application to visual localization. In Conference on Robot Learning.
    """
    n = log_ws.shape[0]
    log_ws_q = jnp.concatenate([log_ws[:, None], jnp.full((n, 1), jnp.log((1 - alpha) / n))], axis=-1)
    log_ws_q = jax.scipy.special.logsumexp(log_ws_q, axis=-1)
    inds = _multinomial(key, log_ws_q)
    log_ws_post = log_ws - log_ws_q
    return log_ws_post - jax.scipy.special.logsumexp(log_ws_post), samples[inds]


def gumbel_softmax(key: JKey, log_ws, samples: JArray, tau: float) -> Tuple[JArray, JArray]:
    """Gumbel-softmax resampling.

    Parameters
    ----------
    key : JKey
        A JAX random key.
    log_ws : JArray (n, )
        Log weights.
    samples : JArray (n, ...)
        Particles.
    tau : float
        The temperature parameter, must be positive. Approaching -> 0 means multinomial resampling.

    Returns
    -------
    (n, ), (n, d)
        New log weights and particles.

    References
    ----------
    Jang, Eric, Shixiang Gu, and Ben Poole.
    Categorical Reparametrization with Gumble-Softmax. ICLR 2017.
    """
    n = log_ws.shape[0]

    def one_sample(key_):
        us = jax.random.uniform(key_, minval=0., maxval=1., shape=(n,))
        gs = -jnp.log(-jnp.log(us))
        return jax.nn.softmax((gs + log_ws) / tau) @ samples

    keys = jax.random.split(key, num=n)
    return jnp.full((n,), -jnp.log(n)), jax.vmap(one_sample)(keys)


def diffusion_resampling(key: JKey, log_ws: JArray, samples: JArray, a: float, ts: JArray,
                         integrator: str = 'euler',
                         ode: bool = True) -> Tuple[JArray, JArray]:
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
    ode : bool
        If True, use the probability flow ODE.

    Returns
    -------
    (n, ), (n, d)
        New log weights and particles.

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
        if ode:
            return a * mu + 0.5 * b2 * jax.vmap(s, in_axes=[0, None])(x, T - t)
        else:
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
            m, scale = euler_maruyama(drift, 0. if ode else b2 ** 0.5, x, t_km1, dt)
        elif integrator == 'lord_and_rougemont':
            m, scale = lord_and_rougemont(-a, f, 0. if ode else b2 ** 0.5, x, t_km1, dt)
        elif integrator == 'jentzen_and_kloeden':
            m, scale = jentzen_and_kloeden(-a, f, 0. if ode else b2 ** 0.5, x, t_km1, dt)
        elif integrator == 'diffrax' and ode:
            term = ODETerm(lambda t_, x_, args: drift(x_, t_))
            m = diffeqsolve(term, Euler(), t0=0., t1=T, dt0=T / (ts.shape[0] - 1), y0=xTs).ys[0]
            scale = 0.
        else:
            raise ValueError(f'Unknown integrator {integrator}.')
        return m + scale * rnd, None

    key, _ = jax.random.split(key)
    keys = jax.random.split(key, num=nsteps)
    x0s, _ = jax.lax.scan(scan_body, xTs, (ts[:-1], ts[1:], keys))
    return jnp.full((n,), -jnp.log(n)), x0s

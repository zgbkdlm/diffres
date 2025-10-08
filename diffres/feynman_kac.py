"""
Generic Feynman--Kac models.
"""
import math
import jax
import jax.numpy as jnp
from diffres.tools import leading_concat
from diffres.typings import JArray, JKey, PyTree, JFloat
from typing import Callable, Tuple, Union, NamedTuple, Optional


class FeynmanKac(NamedTuple):
    m0_log_g0: Tuple[Union[Callable[[JArray], JArray], JArray], Callable[[JArray], JArray]]
    m_log_g: Tuple[Callable[[JArray, JArray, PyTree], JArray], Callable[[JArray, JArray, PyTree], JArray]]
    log_m: Optional[Callable[[JArray, JArray, PyTree], JArray]] = None


def smc_feynman_kac(key: JKey,
                    m0: Union[Callable[[JArray], JArray], JArray],
                    log_g0: Callable[[JArray], JArray],
                    m_log_g: Callable[[JArray, JArray, PyTree], Tuple[JArray, JArray]],
                    scan_pytree: PyTree,
                    nparticles: int,
                    nsteps: int,
                    resampling: Callable[[JKey, JArray, JArray], Tuple[JArray, JArray]],
                    resampling_threshold: float = 1.,
                    return_path: bool = False) -> Tuple[JArray, JArray, JFloat, JArray]:
    r"""Sequential Monte Carlo simulation of a Feynman--Kac model.

    .. math::

        Q_{0:N}(u_{0:N}) \propto M0(u0) \, G0(u0) \prod_{n=1}^N M_{n \mid n-1}(u_n \mid u_{n-1}) \, G_n(u_n, u_{n-1})

    Parameters
    ----------
    key : JKey
        A JAX random key.
    m0 : Callable [JKey -> (s, ...)] or Array [(s, ...)]
        The initial sampler that draws s independent samples. Or, it can also be an array of samples.
    log_g0 : Callable [(s, ...) -> (s, )]
        The initial (log) potential function. Given `s` samples, this function should return `s` weights in an array.
    m_log_g : Callable [JKey, (s, ...), PyTree -> (s, ), (s, ...)]
        The Markov proposal and log potential at time k.
        From left to right, the arguments are, key, input samples, and a pytree parameter.
        The output is a pair of arrays with leading axis of the input samples array.
        The first output is the (unnormalised log) weights, followed by the proposed samples.
    scan_pytree : PyTree
        A PyTree container that is going to be scanned over scan steps. The elements should have a consistent leading
        axis of size `N`. This container will be a tree-parameter input to the transition kernel and the potential function.
    nparticles : int
        The number of particles `s`.
    nsteps : int
        The number of time steps `N`.
    resampling : Callable [JKey, (s, ), (s, ...) -> (s, ), (s, ...)]
        The resampling scheme. Given a tuple of JKey, log weights, and samples, this function should return a pair of
        resampled log weights and samples.
    resampling_threshold : float, default=1.
        The threshold of ESS for resampling. If the current ESS < threshold * N, then apply resampling. Default is 1
        meaning resampling at every step.
    return_path : bool, default=False
        Set True to return all the historical particles.

    Returns
    -------
    JArrays [(N + 1, s, ...), (N + 1, s), (N + 1, )] or [(s, ...), (s, ), (N + 1, )]
    A tuple of three arrays. If `return_path` then the return sizes of the arrays are
    `(N + 1, s, ...), (N + 1, s), (N + 1, )`. Else are (s, ...), (s, ), (N + 1, ).
    """
    key_init, key_body = jax.random.split(key)

    if callable(m0):
        samples0 = m0(key_init)
    else:
        samples0 = m0
    log_ws0_ = log_g0(samples0)
    c = jax.scipy.special.logsumexp(log_ws0_)
    log_ws0 = log_ws0_ - c
    nll0 = -c
    ess0 = compute_ess(log_ws0)

    def scan_body(carry, elem):
        samples, log_ws, nll_, ess = carry
        pytree, key_k = elem
        key_resample, key_markov = jax.random.split(key_k)

        log_ws, samples = jax.lax.cond(ess < resampling_threshold * nparticles,
                                       lambda _: (resampling(key_resample, log_ws, samples)),
                                       lambda _: (log_ws, samples),
                                       None)
        log_ws_, prop_samples = m_log_g(key_markov, samples, pytree)
        log_ws = log_ws + log_ws_

        c_ = jax.scipy.special.logsumexp(log_ws)
        log_ws = log_ws - c_
        nll_ = nll_ - c_
        ess = compute_ess(log_ws)

        return (prop_samples, log_ws, nll_, ess), (prop_samples, log_ws, ess) if return_path else ess

    keys = jax.random.split(key_body, num=nsteps)
    if return_path:
        (_, _, nll, _), (sampless, log_wss, esss) = jax.lax.scan(scan_body,
                                                                 (samples0, log_ws0, nll0, ess0),
                                                                 (scan_pytree, keys))
        return leading_concat(samples0, sampless), leading_concat(log_ws0, log_wss), nll, leading_concat(ess0, esss)
    else:
        (samplesN, log_wsN, nll, _), esss = jax.lax.scan(scan_body,
                                                         (samples0, log_ws0, nll0, ess0),
                                                         (scan_pytree, keys))
        return samplesN, log_wsN, nll, leading_concat(ess0, esss)


def compute_ess(log_ws: JArray) -> JArray:
    """Effective sample size.
    """
    return jnp.exp(-jax.scipy.special.logsumexp(log_ws * 2))

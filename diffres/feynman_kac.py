"""
Generic Feynman--Kac models.
"""
import math
import jax
import jax.numpy as jnp
from diffres.tools import leading_concat
from diffres.typings import JArray, JKey, PyTree
from typing import Callable, Tuple, Union, NamedTuple, Optional


class FeynmanKac(NamedTuple):
    m0: Union[Callable[[JArray], JArray], JArray]
    log_g0: Callable[[JArray], JArray]
    m: Callable[[JArray, JArray, PyTree], JArray]
    log_g: Callable[[JArray, JArray, PyTree], JArray]
    log_m: Optional[Callable[[JArray, JArray, PyTree], JArray]] = None


class CSMCState(NamedTuple):
    path: JArray
    lineage: JArray


def smc_feynman_kac(key: JKey,
                    m0: Union[Callable[[JArray], JArray], JArray],
                    log_g0: Callable[[JArray], JArray],
                    m: Callable[[JArray, JArray, PyTree], JArray],
                    log_g: Callable[[JArray, JArray, PyTree], JArray],
                    scan_pytree: PyTree,
                    nparticles: int,
                    nsteps: int,
                    resampling: Callable[[JKey, JArray], JArray],
                    resampling_threshold: float = 1.,
                    return_path: bool = False) -> Tuple[JArray, JArray, JArray]:
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
    m : Callable [JKey, (s, ...), PyTree -> (s, ...)]
        The Markov transition kernel at k. From left to right, the arguments are, key, input samples, and a pytree
        parameter. The output is an array of the leading axis of the input samples array.
    log_g : Callable [(s, ...), (s, ...), PyTree -> (s, )]
        The (log) potential function. From left to right, the arguments are for `u_n`, `u_{n-1}`, and a pytree
        parameter. The output is an array of size `s`.
    scan_pytree : PyTree
        A PyTree container that is going to be scanned over scan steps. The elements should have a consistent leading
        axis of size `N`. This container will be a tree-parameter input to the transition kernel and the potential function.
    nparticles : int
        The number of particles `s`.
    nsteps : int
        The number of time steps `N`.
    resampling : Callable [JKey, (s, ) -> (s, )]
        The resampling scheme. Given a pair of JKey and weights, this function should return an array of indices for
        the resampling.
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

    TODO
    ----
    The computation for M and G should be merged together.
    """
    key_init, key_body = jax.random.split(key)
    flat_log_ws = -math.log(nparticles) * jnp.ones(nparticles)

    if callable(m0):
        samples0 = m0(key_init)
    else:
        samples0 = m0
    log_ws0_ = log_g0(samples0)
    log_ws0 = log_ws0_ - jax.scipy.special.logsumexp(log_ws0_)
    ess0 = compute_ess(log_ws0)

    def scan_body(carry, elem):
        samples, log_ws, ess = carry
        pytree, key_k = elem
        key_resample, key_markov = jax.random.split(key_k)

        samples, log_ws = jax.lax.cond(ess < resampling_threshold * nparticles,
                                       lambda _: (samples[resampling(key_resample, jnp.exp(log_ws))], flat_log_ws),
                                       lambda _: (samples, log_ws),
                                       None)

        prop_samples = m(key_markov, samples, pytree)
        log_ws_ = log_ws + log_g(prop_samples, samples, pytree)
        log_ws = log_ws_ - jax.scipy.special.logsumexp(log_ws_)
        ess = compute_ess(log_ws)

        return (prop_samples, log_ws, ess), (prop_samples, log_ws, ess) if return_path else ess

    keys = jax.random.split(key_body, num=nsteps)
    if return_path:
        _, (sampless, log_wss, esss) = jax.lax.scan(scan_body, (samples0, log_ws0, ess0), (scan_pytree, keys))
        return leading_concat(samples0, sampless), leading_concat(log_ws0, log_wss), leading_concat(ess0, esss)
    else:
        (samplesN, log_wsN, _), esss = jax.lax.scan(scan_body, (samples0, log_ws0, ess0), (scan_pytree, keys))
        return samplesN, log_wsN, leading_concat(ess0, esss)


def csmc_kernel(key: JKey,
                state: CSMCState,
                fk_model: FeynmanKac,
                scan_pytree: PyTree,
                cond_resampling: Callable[[JKey, JArray, int, int], JArray],
                bwd_method: str = 'tracing') -> Tuple[CSMCState, JArray]:
    """CSMC MCMC kernel.

    Parameters
    ----------
    key : JKey
        Random number generator key.
    state : CSMCState
        The current state of the CSMC chain.
    fk_model : FeynmanKac
        The Feynman--Kac model.
    scan_pytree : PyTree
        The PyTree parameter for the CSMC algorithm. The leading axis of the PyTree should be `nsteps`.
    cond_resampling : Callable[[JKey, JArray, int, int], JArray]
        The conditional resampling function.
    bwd_method : str
        The backward sampling method.

    Returns
    -------
    CSMCState, JArray
        The new state of the CSMC chain and the effective sample size.
    """
    key_fwd, key_bwd = jax.random.split(key, 2)
    ref_path, ref_lineage = state

    sampless, log_wss, indss, esss = csmc_fwd(key_fwd, ref_path, ref_lineage, fk_model, scan_pytree, cond_resampling)
    if bwd_method == 'tracing':
        # TODO: The backward sampling is not necessary for the random walk version.
        ref_path, lineage = csmc_bwd_tracing(key_bwd, sampless, log_wss, indss)
    elif bwd_method == 'sampling':
        ref_path, lineage = csmc_bwd_sampling(key_bwd, sampless, log_wss, fk_model, scan_pytree)
    else:
        raise NotImplementedError(f'{bwd_method} not implemented.')
    return CSMCState(ref_path, lineage), esss


def csmc_fwd(key: JKey,
             ref_path: JArray,
             ref_lineage: JArray,
             fk_model: FeynmanKac,
             scan_pytree: PyTree,
             cond_resampling: Callable[[JKey, JArray, int, int], JArray]) -> Tuple[JArray, JArray, JArray, JArray]:
    """The forward filtering pass of the CSMC chain.

    t0, t1, ..., tT
    x0, x1, ..., xT
    y0, y1, ..., yT

    Parameters
    ----------
    key : JKey
        Random number generator key.
    ref_path : JArray (nsteps + 1, ...)
        The CSMC reference path. The leading axis should be `nsteps + 1`.
    ref_lineage : JArray (nsteps + 1, )
        The CSMC reference lineage. The leading axis should be `nsteps + 1`.
    fk_model : FeynmanKac
        The Feynman--Kac model.
    scan_pytree : PyTree
        The PyTree parameter for the CSMC algorithm; can be, e.g., the measurements.
    cond_resampling : Callable[[JKey, JArray, int, int], JArray]
        The conditional resampling function.

    Returns
    -------
    JArray (nsteps + 1, s, ...), JArray (nsteps + 1, s), JArray (nsteps, ), JArray (nsteps + 1, )
        The CSMC filtering samples, the CSMC filtering samples' weights, the ancestors, and the effective sample sizes.
    """
    nsteps = ref_lineage.shape[0] - 1
    m0, log_g0, m, log_g, _ = fk_model
    key_init, key_body = jax.random.split(key)

    if callable(m0):
        samples0 = m0(key_init)
    else:
        samples0 = m0
    samples0.at[ref_lineage[0]].set(ref_path[0])

    log_ws0_ = log_g0(samples0)
    log_ws0 = log_ws0_ - jax.scipy.special.logsumexp(log_ws0_)
    ess0 = compute_ess(log_ws0)

    def scan_body(carry, elem):
        samples, log_ws = carry
        ref_km1, ref_k, lineage_km1, lineage_k, pytree_k, key_k = elem
        key_resample, key_markov = jax.random.split(key_k)

        inds = cond_resampling(key_resample, jnp.exp(log_ws), lineage_km1, lineage_k)
        samples = samples[inds]

        prop_samples = m(key_markov, samples, pytree_k)
        prop_samples = prop_samples.at[lineage_k].set(ref_k)

        log_ws_ = log_g(prop_samples, samples, pytree_k)
        log_ws = log_ws_ - jax.scipy.special.logsumexp(log_ws_)
        ess = compute_ess(log_ws)
        return (prop_samples, log_ws), (prop_samples, log_ws, inds, ess)

    keys = jax.random.split(key_body, num=nsteps)

    scan_elems = (ref_path[:-1], ref_path[1:], ref_lineage[:-1], ref_lineage[1:], scan_pytree, keys)
    _, (sampless, log_wss, indss, esss) = jax.lax.scan(scan_body, (samples0, log_ws0), scan_elems)
    return leading_concat(samples0, sampless), leading_concat(log_ws0, log_wss), indss, leading_concat(ess0, esss)


def csmc_bwd_tracing(key, sampless, log_wss, indss) -> Tuple[JArray, JArray]:
    """Backward sampling (ancestor tracing) for the CSMC algorithm.

    Parameters
    ----------
    key : JKey
        Random number generator key.
    sampless : JArray (nsteps + 1, s, ...)
        The CSMC filtering samples.
    log_wss : JArray (nsteps + 1, s)
        The CSMC filtering samples' weights.
    indss : JArray (nsteps, )
        The ancestors.

    Returns
    -------
    JArray (nsteps + 1, ...), JArray (nsteps + 1, )
        The CSMC reference path and the reference lineage.
    """
    log_wsT = log_wss[-1]
    n = log_wsT.shape[0]

    bT = jax.random.choice(key, n, (), p=jnp.exp(log_wsT))
    ref_pathT = sampless[-1, bT]

    def scan_body(carry, elem):
        bk = carry
        inds_k, samples_km1 = elem

        b_km1 = inds_k[bk]
        ref_path_km1 = samples_km1[b_km1]
        return b_km1, (ref_path_km1, b_km1)

    _, (ref_path, bs) = jax.lax.scan(scan_body, bT, (indss, sampless[:-1]), reverse=True)
    return leading_concat(ref_path, ref_pathT), leading_concat(bs, bT)


def csmc_bwd_sampling(key, sampless, log_wss, fk_model, scan_pytree) -> Tuple[JArray, JArray]:
    """Backward sampling for the CSMC algorithm.

    Parameters
    ----------
    key : JKey
        Random number generator key.
    sampless : JArray (nsteps + 1, s, ...)
        The CSMC filtering samples.
    log_wss : JArray (nsteps + 1, s)
        The CSMC filtering samples' weights.
    fk_model : FeynmanKac
        The Feynman--Kac model.
    scan_pytree : PyTree
        The PyTree parameter for the CSMC algorithm; can be, e.g., the measurements.

    Returns
    -------
    JArray (nsteps + 1, ...), JArray (nsteps + 1, )
        The CSMC reference path and the reference lineage.
    """
    log_m, log_g = fk_model.log_m, fk_model.log_g

    log_wsT = log_wss[-1]
    n = log_wsT.shape[0]
    nsteps = log_wss.shape[0] - 1

    bT = jax.random.choice(key, n, (), p=jnp.exp(log_wsT))
    ref_T = sampless[-1, bT]

    def scan_body(carry, elem):
        ref_k = carry
        samples_km1, log_ws_km1, pytree, key_k = elem

        # TODO: double-check
        log_ws = log_m(ref_k, samples_km1, pytree) + log_ws_km1 + log_g(ref_k, samples_km1, pytree)
        log_ws = log_ws - jax.scipy.special.logsumexp(log_ws)

        b_km1 = jax.random.choice(key_k, n, (), p=jnp.exp(log_ws))
        ref_km1 = samples_km1[b_km1]
        return ref_km1, (ref_km1, b_km1)

    keys = jax.random.split(key, num=nsteps)

    _, (ref_path, bs) = jax.lax.scan(scan_body, ref_T, (sampless[:-1], log_wss[:-1], scan_pytree, keys), reverse=True)
    return leading_concat(ref_path, ref_T), leading_concat(bs, bT)


def compute_ess(log_ws: JArray) -> JArray:
    """Effective sample size.
    """
    return jnp.exp(-jax.scipy.special.logsumexp(log_ws * 2))

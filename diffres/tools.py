"""
Tools that I often use in my projects.
"""
import jax
import math
import jax.numpy as jnp
from diffres.typings import JArray, JKey, Array, JFloat
from typing import Callable


def leading_concat(a: JArray, b: JArray) -> JArray:
    """Creating a new leading axis on `a` or `b` and then concat.

    If ndim(a) > ndim(b) then create a new leading axis on `b`.
    """
    if a.ndim > b.ndim:
        return jnp.concatenate([a, b[None, ...]], axis=0)
    else:
        return jnp.concatenate([a[None, ...], b], axis=0)


def scan_prepend(prepend_val, f, init, *args, **kwargs):
    def pre_scan_body():
        pass


def sqrtm(mat: JArray, method: str = 'eigh') -> JArray:
    """Matrix (Hermite) square root.
    """
    if method == 'eigh':
        eigenvals, eigenvecs = jnp.linalg.eigh(mat)
        return eigenvecs @ jnp.diag(jnp.sqrt(eigenvals)) @ eigenvecs.T
    else:
        return jnp.real(jax.scipy.linalg.sqrtm(mat))


def bures(m0, cov0, m1, cov1):
    """The Wasserstein distance between two Gaussians.
    """
    sqrt = sqrtm(cov0)
    A = cov0 + cov1 - 2 * sqrtm(sqrt @ cov1 @ sqrt)
    return jnp.sum((m0 - m1) ** 2) + jnp.trace(A)


def _log_det(chol):
    return 2 * jnp.sum(jnp.log(jnp.abs(jnp.diag(chol))))


def kl(m0, cov0, m1, cov1):
    """KL divergence.
    """
    d = m0.shape[-1]
    chol0 = jax.scipy.linalg.cho_factor(cov0)
    chol1 = jax.scipy.linalg.cho_factor(cov1)
    log_det0 = _log_det(chol0[0])
    log_det1 = _log_det(chol1[0])
    return (jnp.trace(jax.scipy.linalg.cho_solve(chol1, cov0))
            - d + jnp.dot(m1 - m0, jax.scipy.linalg.cho_solve(chol1, m1 - m0))
            + log_det1 - log_det0)


def discretise_lti_sde(A: JArray, gamma: JArray, dt: FloatScalar) -> Tuple[JArray, JArray]:
    """Discretise linear time-invariant SDE
    dX(t) = A X(t) dt + B dW(t), where gamma = B @ B.T,
    to X_{k+1} = F X_k + w_k,
    """
    d = A.shape[0]

    semigroup = jax.scipy.linalg.expm(A * dt)
    phi = jnp.vstack([jnp.hstack([A, gamma]), jnp.hstack([jnp.zeros_like(A), -A.T])])
    AB = jax.scipy.linalg.expm(phi * dt) @ jnp.vstack([jnp.zeros_like(A), jnp.eye(d)])
    cov = AB[0:d, :] @ semigroup.T
    return semigroup, cov


def euler_maruyama(key: JKey, x0: JArray, ts: JArray,
                   drift: Callable, dispersion: Callable,
                   integration_nsteps: int = 1,
                   return_path: bool = False) -> JArray:
    r"""Simulate an SDE using the Euler-Maruyama method.

    Parameters
    ----------
    key : JKey
        JAX random key.
    x0 : JArray (..., )
        Initial value.
    ts : JArray (n + 1, )
        Times :math:`t_0, t_1, \ldots, t_n`.
    drift : Callable (..., ), float -> (..., )
        The drift function.
    dispersion : Callable float -> float
        The dispersion function.
    integration_nsteps : int, default=1
        The number of integration steps between each step.
    return_path : bool, default=False
        Whether return the path or just the terminal value.

    Returns
    -------
    JArray (..., ) or JArray (n + 1, ...)
        The terminal value at :math:`t_n`, or the path at :math:`t_0, \ldots, t_n`.
    """
    keys = jax.random.split(key, num=ts.shape[0] - 1)

    def step(xt, t, t_next, key_):
        def scan_body_(carry, elem):
            x = carry
            rnd, t_ = elem
            x = x + drift(x, t_) * ddt + dispersion(t_) * jnp.sqrt(ddt) * rnd
            return x, None

        ddt = jnp.abs(t_next - t) / integration_nsteps
        rnds = jax.random.normal(key_, (integration_nsteps, *x0.shape))
        return jax.lax.scan(scan_body_, xt, (rnds, jnp.linspace(t, t_next - ddt, integration_nsteps)))[0]

    def scan_body(carry, elem):
        x = carry
        key_, t, t_next = elem

        x = step(x, t, t_next, key_)
        return x, x if return_path else None

    terminal_val, path = jax.lax.scan(scan_body, x0, (keys, ts[:-1], ts[1:]))

    if return_path:
        return leading_concat(x0, path)
    else:
        return terminal_val


def reverse_simulator(key: JKey, u0: JArray, ts: JArray,
                      score: Callable, drift: Callable, dispersion: Callable,
                      integration_nsteps: int = 1,
                      integrator: str = 'euler-maruyama') -> JArray:
    r"""Simulate the time-reversal of an SDE.

    Parameters
    ----------
    key : JKey
        JAX random key.
    u0 : JArray (d, )
        Initial value.
    ts : JArray (n + 1, )
        Times :math:`t_0, t_1, \ldots, t_n`.
    score : Callable (..., d), float -> (..., d)
        The score function
    drift : Callable (d, ), float -> (d, )
        The drift function.
    dispersion : Callable float -> float
        The dispersion function.
    integration_nsteps : int, default=1
        The number of integration steps between each step.
    integrator : str, default='euler-maruyama'
        The integrator for solving the reverse SDE.

    Returns
    -------
    JArray (d, )
        The terminal value of the reverse process at :math:`t_n`.
    """
    T = ts[-1]

    def reverse_drift(u, t):
        return -drift(u, T - t) + dispersion(T - t) ** 2 * score(u, T - t)

    def reverse_dispersion(t):
        return dispersion(T - t)

    if integrator == 'euler-maruyama':
        return euler_maruyama(key, u0, ts, reverse_drift, reverse_dispersion,
                              integration_nsteps=integration_nsteps)
    else:
        raise NotImplementedError(f'Integrator {integrator} not implemented.')


def sampling_gm(key: JKey, ws: Array, ms: Array, eigvals: Array, eigvecs: Array) -> JArray:
    """Sampling a Gaussian mixture distribution.

    Parameters
    ----------
    key : JKey
        A JAX random key.
    ws : Array (n, )
        The weights.
    ms : Array (n, d)
        The means.
    eigvals : Array (n, d)
        The eigenvalues of the covariance matrices.
    eigvecs : Array (n, d, d)
        The eigenvectors of the covariance matrices.

    Returns
    -------
    JAarray (shape, d)
        A sample from the Gaussian mixture distribution.
    """
    n, d = ws.shape[0], ms.shape[1]
    key_cat, key_nor = jax.random.split(key)

    ind = jax.random.choice(key_cat, n, p=ws)
    return ms[ind] + eigvecs[ind] @ (eigvals[ind] ** 0.5 * jax.random.normal(key_nor, (d,)))


def logpdf_mvn(x, m, eigvals, eigvecs):
    """Log pdf of a multivariate Normal distribution (without known eigendecomposition).
    """
    n = m.shape[0]
    res = x - m
    c_ = eigvecs.T @ res
    return -0.5 * (jnp.dot(c_, c_ / eigvals) + jnp.sum(jnp.log(eigvals)) + n * math.log(2 * math.pi))


def logpdf_mvn_chol(x: JArray, m: JArray, chol: JArray) -> JFloat:
    n = m.shape[0]
    z = jax.lax.linalg.triangular_solve(chol, x - m, left_side=True, lower=True)
    log_det = jnp.sum(jnp.log(jnp.diag(chol)))
    return -0.5 * (jnp.dot(z, z) + n * math.log(2 * math.pi) + 2 * log_det)


def logpdf_gm(x, ws, ms, eigvals, eigvecs):
    return jax.scipy.special.logsumexp(jax.vmap(logpdf_mvn, in_axes=[None, 0, 0, 0])(x, ms, eigvals, eigvecs), b=ws)


def chol_solve(chol, b, lower=True):
    return jax.lax.linalg.triangular_solve(chol,
                                           jax.lax.linalg.triangular_solve(chol, b, left_side=True, lower=lower,
                                                                           transpose_a=not lower),
                                           left_side=True, lower=lower, transpose_a=lower)


def wishart(key, v, n: int, jitter: float = 0.):
    d = v.shape[-1]
    mat = jnp.einsum('ij,', jnp.linalg.cholesky(v) @ jax.random.normal(key, (d, n)))
    return mat @ mat.T + jitter * jnp.eye(d)


def accuracy(predicted_logits: Array, true_labels: Array) -> JFloat:
    """

    Parameters
    ----------
    predicted_logits : Array-like (n, d)
        An array of prediction logits, where `n` and `d` stands for the numbers of data points and classes, resp. The
        logits must sum to one.
    true_labels : Array-like (n, d)
        An array of true labels in one-hot encoding.

    Returns
    -------
    JFloat
        The accuracy.
    """
    predicted_labels = jnp.argmax(predicted_logits, axis=-1)
    n_corrected_preds = jnp.sum(jnp.diag(true_labels[:, predicted_labels]))
    return n_corrected_preds / predicted_labels.shape[0]


def sum_except_leading(arr: Array) -> JArray:
    """Sum over an array except for the leading axis.

    Parameters
    ----------
    arr : (n, ...) or (n, )
        When the array's ndim is larger than 1, then all axes except for the leading one are summed.
        However, when the array is 1-dimensional, this function reduces to a standard sum.

    Returns
    -------
    JArray (n, ) or ()
        The sum of the array except for the leading axis.
    """
    if arr.ndim == 1:
        return jnp.sum(arr)
    else:
        return jnp.sum(jnp.reshape(arr, (arr.shape[0], -1)), axis=-1)

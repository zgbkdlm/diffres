import jax
import jax.numpy as jnp
from diffres.typings import FloatScalar, JArray
from typing import Callable, Tuple


def euler_maruyama(drift, b, x, t, dt) -> Tuple[JArray, JArray]:
    """
    dX(t) = f(X(t), t) dt + b dW(t)

    Parameters
    ----------
    drift : (d, ), () -> (d, )
        The drift function.
    dispersion : () -> ()
        The dispersion function.
    x : JArray (d, )
        The current state.
    t : float
        The current time.
    dt : float
        The time interval.

    Returns
    -------
    (d, ), (d, )
        Mean and scale for the move.
    """
    return x + drift(x, t) * dt, dt ** 0.5 * b


def lord_and_rougemont(a: FloatScalar,
                       f: Callable[[JArray, FloatScalar], JArray],
                       b: FloatScalar,
                       x: JArray,
                       t: FloatScalar,
                       dt: FloatScalar) -> Tuple[JArray, JArray]:
    """
    dX(t) = a X(t) + f(X(t), t) dt + b dW(t)

    Parameters
    ----------
    a : JArray (d, ) or float
        The linear coefficient.
    f : (d, ), () -> (d, )
        The non-linear function.
    b : JArray (d, ) or float
        The dispersion coefficient.
    x : JArray (d, )
        The current state.
    t : float
        The current time.
    dt : float
        The time interval.

    Returns
    -------
    (d, ), (d, )
        Mean and scale for the move.

    References
    ----------
    Buckwar et al. "The numerical stability of stochastic ordinary differential equations with additive noise."
    Stochastics and Dynamics (2011).
    """
    sg = jnp.exp(dt * a)
    return sg * x + dt * sg * f(x, t), sg * b * dt ** 0.5


def jentzen_and_kloeden(a: FloatScalar,
                        f: Callable[[JArray, FloatScalar], JArray],
                        b: FloatScalar,
                        x: JArray,
                        t: FloatScalar,
                        dt: FloatScalar) -> Tuple[JArray, JArray]:
    """
    dX(t) = a X(t) + f(X(t), t) dt + b dW(t)

    Parameters
    ----------
    a : JArray (d, ) or float
        The linear coefficient.
    f : (d, ), () -> (d, )
        The non-linear function.
    b : JArray (d, ) or float
        The dispersion coefficient.
    x : JArray (d, )
        The current state.
    t : float
        The current time.
    dt : float
        The time interval.

    Returns
    -------
    (d, ), (d, )
        Mean and scale for the move.

    References
    ----------
    Buckwar et al. "The numerical stability of stochastic ordinary differential equations with additive noise."
    Stochastics and Dynamics (2011).

    Notes
    -----
    Bug somewhere could not find.
    """
    sg = jnp.exp(dt * a)
    return sg * x + (sg - 1.) / a * f(x, t), b * ((sg ** 2 - 1.) / (2 * a)) ** 0.5


def tweedie(semigroup, trans_var, score, mu, x, t):
    """Not tested.
    """
    return (x - (1 - semigroup) * mu + trans_var * score(x, t)) / semigroup

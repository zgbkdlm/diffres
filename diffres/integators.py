import jax
import jax.numpy as jnp


def lord_and_rougemont(a, f, b, x, t, dt):
    """
    dX(t) = a X(t) + f(X(t), t) dt + b dW(t)
    Parameters
    ----------
    a
    f
    b
    x
    t
    dt

    Returns
    -------

    """
    sg = jnp.exp(dt * a)
    return sg * x + dt * dt * sg * f(x, t), sg * b


def jentzen_and_kloeden(a, f, b, x, t, dt):
    sg = jnp.exp(dt * a)
    return sg * x + (sg - 1.) / a * f(x, t), b * ((sg ** 2 - 1.) / 2) ** 0.5


def tweedie():
    pass

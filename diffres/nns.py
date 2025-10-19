import os
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from diffres.tools import leading_concat
from diffres.typings import JArray, FloatScalar

kernel_init = nnx.initializers.glorot_uniform()  # constant 0 doesn't work


class NNLoktaVolterra(nnx.Module):
    """A neural network function approximating the state transition.

    Input dimension: (..., 2)
    Output dimension: (..., 2)

    Parameters
    ----------
    dt: the time interval.
    """

    def __init__(self, dt: float, rngs: nnx.Rngs):
        self.dt = dt
        self.linear1 = nnx.Linear(4, 8, kernel_init=kernel_init, rngs=rngs)
        self.act1 = nnx.swish
        self.linear2 = nnx.Linear(8, 16, kernel_init=kernel_init, rngs=rngs)
        self.act2 = nnx.swish
        self.linear3 = nnx.Linear(16, 2, kernel_init=kernel_init, rngs=rngs)

    def __call__(self, x: JArray, dw: JArray):
        if x.shape != dw.shape:
            raise AssertionError('x, dw size must match.')
        z = jnp.concatenate([x, dw], axis=-1)
        return x + self.linear3(self.act2(self.linear2(self.act1(self.linear1(z))))) * self.dt


def nnx_save(model: nnx.Module, filename: str, overwrite: bool = True):
    if not os.path.isabs(filename):
        filename = os.path.abspath(filename)
    _, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(filename, state, force=overwrite)


def nnx_load(model: nnx.Module, filename: str, display: bool = False):
    """Dude, why is Orbax complicating this procedure so much?
    """
    if not os.path.isabs(filename):
        filename = os.path.abspath(filename)
    graphdef, abstract_state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    state_loaded = checkpointer.restore(filename, abstract_state)
    if display:
        nnx.display(state_loaded)
    return nnx.merge(graphdef, state_loaded)

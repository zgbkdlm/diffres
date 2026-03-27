"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
import os
import jax.numpy as jnp
import orbax.checkpoint as ocp
import jax
from flax import nnx
from flax import linen
from jax.flatten_util import ravel_pytree
from diffres.typings import JArray, JKey
from functools import partial
from typing import Sequence, Callable, Iterable, Any, Optional

kernel_init_lokta = nnx.initializers.glorot_uniform(dtype=jnp.float64)  # constant 0 doesn't work
kernel_init_syn = linen.initializers.xavier_normal()


def make_pbnn(nns: Sequence[linen.Module],
              random_argnums: Sequence[int],
              in_dims: Sequence[int],
              batch_size: int,
              keys: JKey) -> tuple[tuple[JArray, Callable], tuple[JArray, Callable], Callable]:
    """
    Parameters
    ----------
    nns : Sequence[linen.Module]
        A sequence (e.g., list and tuple) of neural network instances.
    random_argnums : Sequence[int]
        A sequence of positional integers that specifies the stochastic neural network components. For example, [0, 3]
        means that the first the fourth components in `nns` are the stochastic.
    in_dims : Sequence[int]
        A sequence of integers that define the input dimension of each neural network in `nns`. This in principle
        should be done under the hood of this function, but I don't want to bother to implement it, so please manually
        specify the dimensions.
    batch_size : int
        The data batch size.
    keys : JKey
        JAX random keys. Must be the same length as `nns`.

    Returns
    -------
    Tuple[JArray, Callable], Tuple[JArray, Callable], Callable (dw, ), (dp, ), (n, dx) -> (n, dy),
    Callable (s, dw), (dp, ), (n, dx) -> (s, n, dy)
        Two tuples of initial arrays and pytree functions, a function that evaluates the forward pass of the pBNN, and
        a vmap function of the forward pass.

    Notes
    -----
    The integers in `random_argnums` must be in the ascending order.
    """
    deterministic_argnums = [argnum for argnum in range(len(nns)) if argnum not in random_argnums]

    init_dicts = []
    for nn, in_dim, key in zip(nns, in_dims, keys):
        if isinstance(in_dim, Iterable):
            shape = (batch_size, *in_dim)
        else:
            shape = (batch_size, in_dim)
        init_dicts.append(nn.init(key, jnp.ones(shape)))

    ls_of_random_dicts = [init_dicts[random_argnum] for random_argnum in random_argnums]
    ls_of_deterministic_dicts = [init_dicts[deterministic_argnum] for deterministic_argnum in deterministic_argnums]

    random_array, random_array_to_pytree = ravel_pytree(ls_of_random_dicts)
    deterministic_array, deterministic_array_to_pytree = ravel_pytree(ls_of_deterministic_dicts)

    def forward_pass(random_param: JArray, deterministic_param: JArray, xs: JArray) -> JArray:
        """
        random_param : (dw, )
        deterministic_param : (dp, )
        xs : (n, ...) or (..., )
        return : (n, dy) or (dy, )
        """
        random_param_pytree = random_array_to_pytree(random_param)
        deterministic_param_pytree = deterministic_array_to_pytree(deterministic_param)

        j, k = 0, 0
        out = xs
        for i in range(len(nns)):
            if i in random_argnums:
                out = nns[i].apply(random_param_pytree[j], out)
                j += 1
            else:
                out = nns[i].apply(deterministic_param_pytree[k], out)
                k += 1
        return out

    return (random_array, random_array_to_pytree), (deterministic_array, deterministic_array_to_pytree), forward_pass


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


# NN instances
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
        self.linear1 = nnx.Linear(4, 32, kernel_init=kernel_init_lokta, param_dtype=jnp.float64, rngs=rngs)
        self.act1 = nnx.swish
        self.linear2 = nnx.Linear(32, 2, kernel_init=kernel_init_lokta, param_dtype=jnp.float64, rngs=rngs)

    def __call__(self, x: JArray, dw: JArray):
        if x.shape != dw.shape:
            raise AssertionError('x, dw size must match.')
        z = jnp.concatenate([x, dw], axis=-1)
        return x + self.linear2(self.act1(self.linear1(z))) * self.dt


def pbnn_regression(key, batch_size):
    """Model used for the synthetic regression experiments.
    """

    class NNBlock1(linen.Module):

        @linen.compact
        def __call__(self, x):
            x = linen.Dense(features=50, kernel_init=kernel_init_syn)(x)
            x = linen.gelu(x)
            x = linen.Dense(features=20, kernel_init=kernel_init_syn)(x)
            x = linen.gelu(x)
            return x

    class NNBlock2(linen.Module):

        @linen.compact
        def __call__(self, x):
            x = linen.Dense(features=10, kernel_init=kernel_init_syn)(x)
            x = linen.gelu(x)
            return x

    class NNBlock3(linen.Module):

        @linen.compact
        def __call__(self, x):
            x = linen.Dense(features=20, kernel_init=kernel_init_syn)(x)
            x = linen.gelu(x)
            x = linen.Dense(features=1, kernel_init=kernel_init_syn)(x)
            return x

    input_dims = [1, 20, 10]
    nns = (NNBlock1(), NNBlock2(), NNBlock3())
    random_argnums = (1,)
    keys = jax.random.split(key, num=len(nns))

    pbnn_phi, pbnn_psi, pbnn_forward_pass = make_pbnn(nns, random_argnums, input_dims, batch_size, keys)
    return pbnn_phi, pbnn_psi, pbnn_forward_pass


def pbnn_mnist(key, batch_size):
    class CNNBlock1(linen.Module):

        @linen.compact
        def __call__(self, x):
            x = x.reshape((-1, 28, 28, 1))
            x = linen.Conv(features=32, kernel_size=(3, 3))(x)
            x = linen.relu(x)
            x = linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            return x

    class CNNBlock2(linen.Module):

        @linen.compact
        def __call__(self, x):
            x = linen.Conv(features=64, kernel_size=(3, 3))(x)
            x = linen.relu(x)
            x = linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = x.reshape((x.shape[0], -1))
            x = linen.Dense(features=256)(x)
            x = linen.relu(x)
            x = linen.Dense(features=10)(x)
            return jax.nn.log_softmax(x, axis=-1)

    input_dims = [784, (14, 14, 32)]
    nns = (CNNBlock1(), CNNBlock2())
    random_argnums = (0,)
    keys = jax.random.split(key, num=len(nns))

    pbnn_phi, pbnn_psi, pbnn_forward_pass = make_pbnn(nns, random_argnums, input_dims, batch_size, keys)
    return pbnn_phi, pbnn_psi, pbnn_forward_pass


def pbnn_cifar10(key,
                 batch_size: int,
                 depth: int = 18,
                 group_size: int = 8):
    ModuleDef = Any
    # CIFAR_MEAN = jnp.array([0.4914, 0.4822, 0.4465])
    # CIFAR_STD = jnp.array([0.2023, 0.1994, 0.2010])

    class ResNetBlock(linen.Module):
        filters: int
        conv: ModuleDef
        norm: ModuleDef
        act: Callable
        strides: tuple[int, int] = (1, 1)

        @linen.compact
        def __call__(self, x):
            residual = x
            y = self.conv(self.filters, (3, 3), self.strides)(x)
            y = self.norm()(y)
            y = self.act(y)
            y = self.conv(self.filters, (3, 3))(y)
            y = self.norm(scale_init=linen.initializers.ones_init())(y)

            if residual.shape != y.shape:
                residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(residual)
                residual = self.norm(name="norm_proj")(residual)

            return self.act(residual + y)

    class ResNetHead(linen.Module):
        num_filters: int = 64
        dtype: Any = jnp.float32
        act: Callable = linen.relu
        conv: ModuleDef = linen.Conv

        @linen.compact
        def __call__(self, x):
            x = x.reshape((-1, 32, 32, 3))
            conv = partial(self.conv, use_bias=False, dtype=self.dtype)
            norm = partial(
                linen.GroupNorm,
                num_groups=group_size,
                dtype=self.dtype,
            )

            x = conv(self.num_filters, (3, 3), strides=(1, 1), padding='SAME', name="conv_init")(x)
            x = norm(name="bn_init")(x)
            x = linen.relu(x)
            return x

    class ResNetBody(linen.Module):
        """ResNetV1."""

        stage_sizes: Sequence[int]
        block_cls: ModuleDef
        num_classes: int
        num_filters: int = 64
        dtype: Any = jnp.float32
        act: Callable = linen.relu
        conv: ModuleDef = linen.Conv

        @linen.compact
        def __call__(self, x):
            conv = partial(self.conv, use_bias=False, dtype=self.dtype)
            norm = partial(
                linen.GroupNorm,
                num_groups=group_size,
                dtype=self.dtype,
            )

            for i, block_size in enumerate(self.stage_sizes):
                for j in range(block_size):
                    strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                    x = self.block_cls(
                        self.num_filters * 2 ** i,
                        strides=strides,
                        conv=conv,
                        norm=norm,
                        act=self.act,
                    )(x)
            x = jnp.mean(x, axis=(1, 2))
            x = linen.Dense(self.num_classes, dtype=self.dtype)(x)
            return jax.nn.log_softmax(x, axis=-1)

    configs = [2, 2, 2, 2] if depth == 18 else [3, 4, 6, 3]
    resnet_head = ResNetHead(num_filters=64)
    resnet_body = ResNetBody(stage_sizes=configs, block_cls=ResNetBlock, num_classes=10)
    input_dims = [3072, (32, 32, 64)]
    nns = (resnet_head, resnet_body)
    random_argnums = (0,)
    keys = jax.random.split(key, num=len(nns))

    pbnn_phi, pbnn_psi, pbnn_forward_pass = make_pbnn(nns, random_argnums, input_dims, batch_size, keys)
    return pbnn_phi, pbnn_psi, pbnn_forward_pass

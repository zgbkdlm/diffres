"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
import os
import jax.numpy as jnp
import orbax.checkpoint as ocp
import jax
import shutil
import time
from flax import nnx
from diffres.tools import leading_concat
from diffres.typings import JArray, FloatScalar
from flax import nnx
from diffres.tools import leading_concat
from diffres.typings import JArray, FloatScalar

kernel_init_lokta = nnx.initializers.glorot_uniform(dtype=jnp.float64)  # constant 0 doesn't work
kernel_init = nnx.initializers.glorot_uniform(dtype=jnp.float64)
JArray = jax.Array 

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


class ConvBlock(nnx.Module):
    """Conv → GroupNorm → GELU → Conv → GroupNorm → GELU"""

    def __init__(self, in_channels: int, out_channels: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            padding='SAME',
            rngs=rngs
        )
        self.norm1 = nnx.GroupNorm(
            num_groups=min(8, out_channels),
            num_features=out_channels,
            rngs=rngs
        )
        self.conv2 = nnx.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            padding='SAME',
            rngs=rngs
        )
        self.norm2 = nnx.GroupNorm(
            num_groups=min(8, out_channels),
            num_features=out_channels,
            rngs=rngs
        )

    def __call__(self, x):
        # x: (H, W, C)
        x = self.conv1(x)
        x = self.norm1(x)
        x = nnx.gelu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = nnx.gelu(x)
        return x


class NNWeather_dynamics(nnx.Module):
    """
    UNet dynamics model for weather image state space.

    Takes x_{k-1} of shape (H, W, C) and noise q of shape (H, W, C),
    concatenates them to (H, W, 2C), and predicts x_k of shape (H, W, C).

    Architecture:
        Encoder:     3 levels with strided convolution downsampling
        Bottleneck:  ConvBlock
        Decoder:     3 levels with transposed convolution upsampling + skip connections
        Output:      1x1 conv to C channels + residual + sigmoid
    """

    def __init__(self,
                 img_channels: int = 1,
                 channel_widths: list = [32, 64, 128],
                 rngs: nnx.Rngs = None):

        c1, c2, c3 = channel_widths
        in_ch = img_channels * 2  # x and q concatenated along channel dim

        # define encoder
        self.enc1 = ConvBlock(in_ch, c1, rngs)  # (H,   W,   c1)
        self.enc2 = ConvBlock(c1,    c2, rngs)  # (H/2, W/2, c2)
        self.enc3 = ConvBlock(c2,    c3, rngs)  # (H/4, W/4, c3)

        # strided convolutions for downsampling
        self.down1 = nnx.Conv(c1, c1, kernel_size=(2, 2), strides=(2, 2), rngs=rngs)
        self.down2 = nnx.Conv(c2, c2, kernel_size=(2, 2), strides=(2, 2), rngs=rngs)

        self.bottleneck = ConvBlock(c3, c3, rngs)  # (H/4, W/4, c3)

        # define decoder
        # transposed convolutions for upsampling
        self.up2 = nnx.ConvTranspose(c3, c2, kernel_size=(2, 2), strides=(2, 2), rngs=rngs)
        self.up1 = nnx.ConvTranspose(c2, c1, kernel_size=(2, 2), strides=(2, 2), rngs=rngs)

        self.dec2 = ConvBlock(c2 + c2, c2, rngs) 
        self.dec1 = ConvBlock(c1 + c1, c1, rngs)

        self.out_conv = nnx.Conv(
            in_features=c1,
            out_features=img_channels,
            kernel_size=(1, 1),
            padding='SAME',
            rngs=rngs
        )

    def __call__(self, x, q):
        """
        x: (H, W, C)   current particle/state image
        q: (H, W, C)   noise image
        returns: (H, W, C) next state image, values in (0, 1)
        """
        # concatenate state and noise along channel dim
        xq = jnp.concatenate([x, q], axis=-1)  # (H, W, 2C)

        # encoder
        s1 = self.enc1(xq)              # (H,   W,   c1), skip connection 1
        s2 = self.enc2(self.down1(s1))  # (H/2, W/2, c2), skip connection 2
        s3 = self.enc3(self.down2(s2))  # (H/4, W/4, c3)

        b = self.bottleneck(s3)         # (H/4, W/4, c3)

        # decoder
        d2 = self.up2(b)                         # (H/2, W/2, c2)
        d2 = jnp.concatenate([d2, s2], axis=-1)  # (H/2, W/2, c2+c2)
        d2 = self.dec2(d2)                       # (H/2, W/2, c2)

        d1 = self.up1(d2)                        # (H,   W,   c1)
        d1 = jnp.concatenate([d1, s1], axis=-1)  # (H,   W,   c1+c1)
        d1 = self.dec1(d1)                       # (H,   W,   c1)

        delta = self.out_conv(d1)        # (H, W, C)

        return jax.nn.sigmoid(x + delta) # (H, W, C)

    def f_dynamics(self, x, q):
        return self.__call__(x, q)


_CHECKPOINTER = ocp.StandardCheckpointer()

def nnx_save(model: nnx.Module, filename: str, overwrite: bool = True):
    if not os.path.isabs(filename):
        filename = os.path.abspath(filename)

    tmp = filename + '.orbax-checkpoint-tmp'

    def secure_delete(path):
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except PermissionError:
                time.sleep(0.5)
                try:
                    shutil.rmtree(path)
                except Exception as e:
                    print(f'Warning: could not remove {path} after retry: {e}')

    secure_delete(filename)
    secure_delete(tmp)

    _, state = nnx.split(model)

    try:
        _CHECKPOINTER.save(filename, state, force=overwrite)
        _CHECKPOINTER.wait_until_finished()
    except Exception as e:
        print(f'Warning: checkpoint save failed for {filename}: {e}. Training continues.')



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

import os
import jax.numpy as jnp
import orbax.checkpoint as ocp
import jax
from flax import nnx
from diffres.tools import leading_concat
from diffres.typings import JArray, FloatScalar

kernel_init = nnx.initializers.glorot_uniform(dtype=jnp.float64)
kernel_init_lokta = nnx.initializers.glorot_uniform(dtype=jnp.float64)
JArray = jax.Array 

def siren_init_first_layer(key, shape, dtype=jnp.float64):
    """SIREN init for the initial layer."""
    in_dim = shape[0]
    max_val = 1.0 / in_dim
    return jax.random.uniform(key, shape, dtype, minval=-max_val, maxval=max_val)

def siren_init_hidden_layer(key, shape, dtype=jnp.float64, w0=30):
    """SIREN init for hidden layers."""
    in_dim = shape[0]
    max_val = jnp.sqrt(6.0 / in_dim) / w0
    return jax.random.uniform(key, shape, dtype, minval=-max_val, maxval=max_val)

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

class NNPendulum_ForwardDynamics(nnx.Module):
    """A neural network function approximating the state transition f(x,q).
    The network is designed based on the SIREN architecture.

    Input dimension: (..., 2)
    Output dimension: (..., 2)

    Parameters
    ----------
    dt: the time interval.
    """

    def __init__(self, dt: float, rngs: nnx.Rngs):
        self.dt = dt
        input_dim = 5
        hidden_dim = 256
        self.w0 = 8.0
        self.vel_scale = 10.
        self.linear1 = nnx.Linear(input_dim, hidden_dim, kernel_init=siren_init_first_layer, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, kernel_init=lambda k, s, d=jnp.float64: siren_init_hidden_layer(k, s, d, w0=self.w0), rngs=rngs)
        self.linear3 = nnx.Linear(hidden_dim, hidden_dim, kernel_init=lambda k, s, d=jnp.float64: siren_init_hidden_layer(k, s, d, w0=self.w0), rngs=rngs)
        self.linear4 = nnx.Linear(hidden_dim, 2, kernel_init=kernel_init, use_bias=False, rngs=rngs)

        #self.vel_min = -2
        #self.vel_max = 2
        #self.vel_range = (self.vel_max - self.vel_min) + 1e-6

    def __call__(self, x: JArray, q: JArray):
        if x.shape != q.shape:
            raise AssertionError('State x and noise q must have the same size.')
        angle = x[..., 0:1]
        ang_vel = x[..., 1:2]
        ang_vel_normalized = ang_vel / self.vel_scale
        #ang_vel_normalized_01 = (ang_vel - self.vel_min) / self.vel_range
        #ang_vel_normalized = 2.0 * ang_vel_normalized_01 - 1.0
        x_embedded = jnp.concatenate(
            [jnp.cos(angle), jnp.sin(angle), ang_vel_normalized], 
            axis=-1)
        z = jnp.concatenate([x_embedded, q], axis=-1)
        h = jnp.sin(self.w0*self.linear1(z))
        h = jnp.sin(self.linear2(h))
        h = jnp.sin(self.linear3(h))
        delta_x = self.linear4(h)
        return x + delta_x * self.dt

def fourier_embed(x, n_freqs=10):
    """Construct high-frequency features."""
    freqs = 2.0 ** jnp.arange(n_freqs)
    embedded_features = [x]
    for freq in freqs:
        embedded_features.append(jnp.sin(freq * x))
        embedded_features.append(jnp.cos(freq * x))
    return jnp.concatenate(embedded_features, axis=-1)

class NNPendulum_ForwardDynamics_fourier(nnx.Module):
    """A neural network function approximating the state transition f(x,q).

    Input dimension: (..., 2)
    Output dimension: (..., 2)

    Parameters
    ----------
    dt: the time interval.
    """

    def __init__(self, dt: float, rngs: nnx.Rngs):
        self.dt = dt
        self.n_freqs = 2
        original_input_dim = 4
        embedded_dim = original_input_dim + (original_input_dim * self.n_freqs * 2)
        self.linear1 = nnx.Linear(embedded_dim, 64, kernel_init=kernel_init, rngs=rngs)
        self.act1 = nnx.tanh
        self.linear2 = nnx.Linear(64, 64, kernel_init=kernel_init, rngs=rngs)
        self.act2 = nnx.tanh
        self.linear3 = nnx.Linear(64, 64, kernel_init=kernel_init, rngs=rngs)
        self.act3 = nnx.tanh
        self.linear4 = nnx.Linear(64, 2, kernel_init=kernel_init, rngs=rngs)

    def __call__(self, x: JArray, q: JArray):
        if x.shape != q.shape:
            raise AssertionError('State x and noise q must have the same size.')
        z = jnp.concatenate([x, q], axis=-1)
        z_embedded = fourier_embed(z, n_freqs=self.n_freqs)
        h = self.act1(self.linear1(z_embedded))
        h = self.act2(self.linear2(h))
        h = self.act3(self.linear3(h))
        delta_x = self.linear4(h)
        return x + delta_x

class NNPendulum_decoder_unet(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        start_channels = 16  
        self.linear1 = nnx.Linear(2, 16, rngs=rngs)
        self.linear2 = nnx.Linear(16, 4 * 4 * start_channels, rngs=rngs)
        self.start_channels = start_channels
        self.up_block1 = self._create_residual_block(start_channels, 16, rngs) 
        self.up_block2 = self._create_residual_block(16, 8, rngs) 
        self.up_block3 = self._create_residual_block(8, 8, rngs) 
        self.final_norm = nnx.LayerNorm(8, rngs=rngs) 
        self.final_conv = nnx.Conv(in_features=8, 
                                 out_features=1,
                                 kernel_size=(3, 3),
                                 padding='SAME',
                                 rngs=rngs)

    def _create_residual_block(self, 
                              in_channels: int, 
                              out_channels: int, 
                              rngs: nnx.Rngs):
        conv1 = nnx.Conv(in_features=in_channels, 
                        out_features=out_channels, 
                        kernel_size=(3, 3), 
                        padding='SAME', 
                        rngs=rngs)
        norm1 = nnx.LayerNorm(out_channels, rngs=rngs)
        conv2 = nnx.Conv(in_features=out_channels, 
                        out_features=out_channels, 
                        kernel_size=(3, 3), 
                        padding='SAME', 
                        rngs=rngs)
        norm2 = nnx.LayerNorm(out_channels, rngs=rngs)
        shortcut = nnx.Conv(in_features=in_channels, 
                            out_features=out_channels, 
                            kernel_size=(1, 1), 
                            rngs=rngs)

        def forward(x: jax.Array) -> jax.Array:
            identity = shortcut(x)
            x = conv1(x)
            x = norm1(x)
            x = nnx.gelu(x)
            x = conv2(x)
            x = norm2(x)
            x = nnx.gelu(x)
            return x + identity
        return forward

    def _upsample(self, x: jax.Array, target_size: int) -> jax.Array:
        return jax.image.resize(x, 
                              (x.shape[0], target_size, target_size, x.shape[3]), 
                              method='nearest')

    def __call__(self, x: JArray):
        z = nnx.gelu(self.linear1(x))
        z = nnx.gelu(self.linear2(z))
        batch_dims = x.shape[:-1]
        z = z.reshape((*batch_dims, 4, 4, self.start_channels))
        z = self._upsample(z, 8)
        z = self.up_block1(z) 
        z = self._upsample(z, 16)
        z = self.up_block2(z) 
        z = self._upsample(z, 32)
        z = self.up_block3(z)
        z = self.final_norm(z)
        z = nnx.gelu(z)
        z = self.final_conv(z)
        return z
    
class NNPendulum_decoder(nnx.Module):
    """
    A neural network (decoder) approximating the image observation model g(x).

    Input dimension: (..., 2) 
    Output dimension: (..., 32, 32, 1)
    """
    def __init__(self, *, rngs: nnx.Rngs):
        input_features = 3
        self.vel_scale = 10.
        self.linear1 = nnx.Linear(input_features, 16, rngs=rngs)
        self.act1 = nnx.relu
        start_channels = 16
        self.linear2 = nnx.Linear(16, 4 * 4 * start_channels, rngs=rngs) 
        self.act2 = nnx.relu
                
        self.deconv1 = nnx.ConvTranspose(
            in_features=start_channels,  
            out_features=32,           
            kernel_size=(3, 3), 
            strides=(2, 2), 
            padding='SAME', 
            rngs=rngs
        )
        self.act3 = nnx.relu
        
        self.deconv2 = nnx.ConvTranspose(
            in_features=32,             
            out_features=16,            
            kernel_size=(3, 3), 
            strides=(2, 2), 
            padding='SAME', 
            rngs=rngs
        )
        self.act4 = nnx.relu
        
        self.deconv3 = nnx.ConvTranspose(
            in_features=16,            
            out_features=1,            
            kernel_size=(3, 3), 
            strides=(2, 2), 
            padding='SAME', 
            rngs=rngs
        )

        #self.vel_min = -2
        #self.vel_max = 2
        #self.vel_range = (self.vel_max - self.vel_min) + 1e-6

    def __call__(self, x: JArray):
        alpha = x[..., 0:1]
        alpha_dot = x[..., 1:2]
        alpha_dot_normalized = alpha_dot / self.vel_scale
        sin_angle = jnp.sin(alpha)
        cos_angle = jnp.cos(alpha)

        #alpha_dot_normalized_01 = (alpha_dot - self.vel_min) / self.vel_range
        #alpha_dot_normalized = 2.0 * alpha_dot_normalized_01 - 1.0

        z = jnp.concatenate([
            sin_angle, 
            cos_angle, 
            alpha_dot_normalized], axis=-1)

        z = self.act1(self.linear1(z))
        z = self.act2(self.linear2(z))
        batch_dims = x.shape[:-1]
        z = z.reshape((*batch_dims, 4, 4, 16)) 
        z = self.act3(self.deconv1(z))
        z = self.act4(self.deconv2(z))
        z = self.deconv3(z)
        return z

class NNPendulum_decoder_SSM(nnx.Module):
    """Container model holding both the NN model for the hidden dynamics and the NN observation model (decoder)."""
    def __init__(self, dt: float, *, rngs: nnx.Rngs):
        self.dt = dt
        self.f_dynamics = NNPendulum_ForwardDynamics(dt=dt, rngs=rngs)
        self.g_observation = NNPendulum_decoder(rngs=rngs)

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

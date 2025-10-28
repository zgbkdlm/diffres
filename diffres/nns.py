import os
import jax.numpy as jnp
import orbax.checkpoint as ocp
import jax
from flax import nnx
from diffres.tools import leading_concat
from diffres.typings import JArray, FloatScalar

kernel_init = nnx.initializers.glorot_uniform()


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
    

class Pendulum_1D(nnx.Module):
    def __init__(self, g_init: float, dt: float):
        self.g = nnx.Param(g_init)
        self.dt = dt

    def __call__(self, x, q):
        alpha, ddt_alpha = x
        return jnp.array([alpha + ddt_alpha * self.dt + q[0], ddt_alpha - self.g.value * jnp.sin(alpha) * self.dt + q[1]])


class NNPendulum_ForwardDynamics(nnx.Module):
    """A neural network function approximating the state transition f(x,q).

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

    def __call__(self, x: JArray, q: JArray):
        if x.shape != q.shape:
            raise AssertionError('State x and noise q must have the same size.')
        z = jnp.concatenate([x, q], axis=-1)
        delta_x = self.linear3(self.act2(self.linear2(self.act1(self.linear1(z))))) * self.dt
        return x + delta_x
    

class NNPendulum_ObservationModel(nnx.Module):
    """A neural network approximating the observation mean.

    Input dimension: (..., 2) 
    Output dimension: (..., 1)
    """
    def __init__(self, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(2, 8, kernel_init=kernel_init, rngs=rngs)
        self.act1 = nnx.swish
        self.linear2 = nnx.Linear(8, 8, kernel_init=kernel_init, rngs=rngs)
        self.act2 = nnx.swish
        self.linear3 = nnx.Linear(8, 1, kernel_init=kernel_init, rngs=rngs)

    def __call__(self, x: JArray):
        z = self.linear3(self.act2(self.linear2(self.act1(self.linear1(x)))))
        return z
    

class NNPendulum_SSM(nnx.Module):
    """Container model holding both the NN model for the hidden dynamics and the NN observation model."""
    def __init__(self, dt: float, *, rngs: nnx.Rngs):
        self.dt = dt
        self.f_dynamics = NNPendulum_ForwardDynamics(dt=dt, rngs=rngs)
        self.g_observation = NNPendulum_ObservationModel(rngs=rngs)


class NNPendulum_decoder(nnx.Module):
    """
    A neural network (decoder) approximating the image observation model g(x).

    Input dimension: (..., 2) 
    Output dimension: (..., 32, 32, 1)
    """
    def __init__(self, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(2, 64, rngs=rngs)
        self.act1 = nnx.relu
        start_channels = 16
        self.linear2 = nnx.Linear(64, 4 * 4 * start_channels, rngs=rngs) 
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
        
        self.act5 = nnx.sigmoid 

    def __call__(self, x: JArray):
        z = self.act1(self.linear1(x))
        z = self.act2(self.linear2(z))
        batch_dims = x.shape[:-1]
        z = z.reshape((*batch_dims, 4, 4, 16)) 
        z = self.act3(self.deconv1(z))
        z = self.act4(self.deconv2(z))
        z = self.deconv3(z)
        return self.act5(z)
    

class NNPendulum_decoder2(nnx.Module):
    """
    A neural network (decoder) approximating the image observation model g(x).

    Input dimension: (..., 2) 
    Output dimension: (..., 32, 32, 1)
    ...
    """
    def __init__(self, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(2, 64, rngs=rngs)
        self.act1 = nnx.relu
        start_channels = 16
        self.linear2 = nnx.Linear(64, 4 * 4 * start_channels, rngs=rngs) 
        self.act2 = nnx.relu
                
        self.conv1 = nnx.Conv(
            in_features=start_channels,  
            out_features=32,            
            kernel_size=(3, 3), 
            padding='SAME',
            rngs=rngs
        )
        self.act3 = nnx.relu
        
        self.conv2 = nnx.Conv(
            in_features=32,               
            out_features=16,            
            kernel_size=(3, 3), 
            padding='SAME',
            rngs=rngs
        )
        self.act4 = nnx.relu
        
        self.conv3 = nnx.Conv(
            in_features=16,               
            out_features=1,             
            kernel_size=(3, 3), 
            padding='SAME',
            rngs=rngs
        )

        self.act5 = nnx.sigmoid
    
    def __call__(self, x: JArray):
        z = self.act1(self.linear1(x))
        z = self.act2(self.linear2(z))
        batch_dims = x.shape[:-1]
        z = z.reshape((*batch_dims, 4, 4, 16)) 
        target_shape = (*batch_dims, 8, 8, 16)
        z = jax.image.resize(z, shape=target_shape, method='nearest')
        z = self.act3(self.conv1(z)) 
        target_shape = (*batch_dims, 16, 16, 32)
        z = jax.image.resize(z, shape=target_shape, method='nearest')
        z = self.act4(self.conv2(z)) 
        target_shape = (*batch_dims, 32, 32, 16)
        z = jax.image.resize(z, shape=target_shape, method='nearest')
        z = self.conv3(z) 
        return self.act5(z)
    

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

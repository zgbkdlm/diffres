import jax
import jax.numpy as jnp
import math
from diffres.resampling import diffusion_resampling, stratified, multinomial_stopped, multinomial
import matplotlib.pyplot as plt

jax.config.update('jax_enable_x64', True)

key = jax.random.PRNGKey(666)

nsamples = 100
d = 2


def diff_resampling(key_, log_ws, xs):
    return diffusion_resampling(key_, log_ws, xs,
                                -0.5, jnp.linspace(0., 1., 16), integrator='euler')


def g(x, param):
    """param > 0"""
    return 1 / (1. + 2 * jnp.exp(-jnp.dot(x - param, x - param)))


def sampler(key_, param):
    m = jnp.array([jnp.sin(0.5 * math.pi * param),
                   jnp.cos(0.5 * math.pi * param)])
    log_pot = lambda x: jnp.sum(jax.scipy.stats.norm.logpdf(0., x, 1.), axis=-1)
    xs = m + jax.random.normal(key_, shape=(nsamples, d))
    log_ws = log_pot(xs)
    log_ws = log_ws - jax.scipy.special.logsumexp(log_ws)

    # Resampling
    key, _ = jax.random.split(key_)
    log_ws, xs = diff_resampling(key_, log_ws, xs)
    return log_ws, xs


def loss(key_, param):
    log_ws, xs = sampler(key_, param)
    return jnp.sum(jnp.exp(log_ws) * jax.vmap(g, in_axes=[0, None])(xs, param))


def grad(key_, param):
    return jax.grad(loss, argnums=1)(key_, param)


params = jnp.linspace(-2., 3., 1000)
losses = jax.vmap(loss, in_axes=[None, 0])(key, params)
grads = jax.vmap(grad, in_axes=[None, 0])(key, params)

fig, axes = plt.subplots(ncols=2)
axes[0].plot(params, losses)
axes[1].plot(params, grads)
plt.tight_layout(pad=.1)
plt.show()

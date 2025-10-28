"""This is an example where OT and diffusion should NOT work well.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import math
from diffres.resampling import diffusion_resampling, multinomial, ensemble_ot, gumbel_softmax, soft_resampling
from functools import partial

key = jax.random.PRNGKey(777)
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)

nsamples = 2000


# Model definition
def logpdf_prior(x):
    return jnp.sum(jax.scipy.stats.norm.logpdf(x, 0., 5.), axis=-1)


def logpdf_likelihood(x):
    l1 = jax.scipy.stats.norm.logpdf(1., jnp.linalg.norm(x, ord=2, axis=-1), 0.1)
    l2 = jax.scipy.stats.norm.logpdf(3., jnp.linalg.norm(x, ord=2, axis=-1), 0.1)
    return jax.scipy.special.logsumexp(jnp.array([l1, l2]))


@jax.jit
@partial(jax.vmap, in_axes=[0])
@partial(jax.vmap, in_axes=[0])
def energy(x):
    return jnp.exp(logpdf_prior(x) + logpdf_likelihood(x))


# Plot posterior
ngrids1, ngrids2 = 1024, 1024
grids_p1 = jnp.linspace(-7., 7., ngrids1)
grids_p2 = jnp.linspace(-7., 7., ngrids2)
mgrids = jnp.meshgrid(grids_p1, grids_p2)
cartesian = jnp.dstack(mgrids)

# z = jnp.trapezoid(jnp.trapezoid(energy(cartesian),
#                                 x=grids_p2, axis=0),
#                   x=grids_p1, axis=0)


# Sampling
key, _ = jax.random.split(key)
prior_samples = 5 * jax.random.normal(key, shape=(nsamples, 2))

log_post_ws = jax.vmap(logpdf_likelihood, in_axes=[0])(prior_samples)
log_post_ws = log_post_ws - jax.scipy.special.logsumexp(log_post_ws)
post_ws = jnp.exp(log_post_ws)


# Importance resampling
@jax.jit
def resampling(key_, log_ws_, samples_):
    return diffusion_resampling(key_, log_ws_, samples_,
                                -1., jnp.linspace(0., 2., 512),
                                integrator='euler', ode=False)


# def resampling(key_, log_ws_, samples_):
#     return multinomial(key_, log_ws_, samples_)

# @jax.jit
# def resampling(key_, log_ws_, samples_):
#     return ensemble_ot(key_, log_ws_, samples_, 1 / math.log(nsamples))


key, _ = jax.random.split(key)
_, post_samples = resampling(key, log_post_ws, prior_samples)

plt.contourf(*mgrids, energy(cartesian))
plt.scatter(post_samples[:, 0], post_samples[:, 1], s=1)
plt.show()

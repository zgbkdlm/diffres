"""This is a negative example where both OT and diffusion (with Gaussian ref) should NOT work well.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import math

from IPython.core.pylabtools import figsize

from diffres.resampling import diffusion_resampling, multinomial, ensemble_ot
from functools import partial

key = jax.random.PRNGKey(666)
jax.config.update("jax_enable_x64", True)

nsamples = 1000


# Two-rings model definition
def logpdf_prior(x):
    return jnp.sum(jax.scipy.stats.norm.logpdf(x, 0., 3.), axis=-1)


def logpdf_likelihood(x):
    l1 = jax.scipy.stats.norm.logpdf(1., jnp.linalg.norm(x, ord=2, axis=-1), 0.2)
    l2 = jax.scipy.stats.norm.logpdf(3., jnp.linalg.norm(x, ord=2, axis=-1), 0.2)
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

# Sampling
key, _ = jax.random.split(key)
prior_samples = 3 * jax.random.normal(key, shape=(nsamples, 2))
log_post_ws = jax.vmap(logpdf_likelihood, in_axes=[0])(prior_samples)
log_post_ws = log_post_ws - jax.scipy.special.logsumexp(log_post_ws)
post_ws = jnp.exp(log_post_ws)


# Importance resampling
def resampling_ot(key_, log_ws_, samples_):
    return multinomial(key_, log_ws_, samples_)


def resampling_(key_, log_ws_, samples_):
    return ensemble_ot(key_, log_ws_, samples_, 1 / math.log(nsamples))


key, _ = jax.random.split(key)
_, post_samples_diffusion = diffusion_resampling(key, log_post_ws, prior_samples,
                                                 -1., jnp.linspace(0., 1., 128),
                                                 integrator='jentzen_and_kloeden', ode=True)
_, post_samples_ot = ensemble_ot(key, log_post_ws, prior_samples, eps=0.3)
_, post_samples_multinomial = multinomial(key, log_post_ws, prior_samples)

fig, axes = plt.subplots(figsize=(18, 6), ncols=3, sharey=True)

axes[0].contourf(*mgrids, energy(cartesian), cmap=plt.cm.binary)
axes[0].scatter(post_samples_diffusion[:, 0], post_samples_diffusion[:, 1], s=2,
                edgecolors='none', facecolors='tab:red')
axes[1].contourf(*mgrids, energy(cartesian), cmap=plt.cm.binary)
axes[1].scatter(post_samples_ot[:, 0], post_samples_ot[:, 1], s=2,
                edgecolors='none', facecolors='tab:red')
axes[2].contourf(*mgrids, energy(cartesian), cmap=plt.cm.binary)
axes[2].scatter(post_samples_multinomial[:, 0], post_samples_multinomial[:, 1], s=2,
                edgecolors='none', facecolors='tab:red')
plt.tight_layout(pad=.1)
plt.show()

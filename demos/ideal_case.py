"""
Diffres should exactly recover multinomial resampling at continuous time.
"""
import jax
import jax.numpy as jnp
import numpy as np
from diffres.resampling import multinomial, stratified, systematic, diffusion_resampling, soft_resampling, \
    gumbel_softmax, ensemble_ot
import matplotlib.pyplot as plt


jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(666)

d = 2

nsamples = 64
nsteps = 4096

# Generate dummy data
xs = jax.random.normal(key, shape=(nsamples, d))
pot_fn = lambda x: jnp.sum(jax.scipy.stats.norm.logpdf(-1., x, 0.5), axis=-1)
log_ws = pot_fn(xs)
log_ws = log_ws - jax.scipy.special.logsumexp(log_ws)

ts = jnp.linspace(0., 1., nsteps + 1)

key, _ = jax.random.split(key)
_, samples = diffusion_resampling(key, log_ws, xs, -0.5, ts, integrator='euler', ode=False)

_, m_samples = multinomial(key, log_ws, xs)

plt.scatter(xs[:, 0], xs[:, 1], alpha=.5)
plt.scatter(samples[:, 0], samples[:, 1], alpha=.5)
plt.scatter(m_samples[:, 0], m_samples[:, 1], alpha=.5)
plt.show()

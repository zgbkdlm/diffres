import jax
import jax.numpy as jnp
from diffres.resampling import diffusion_resampling
import matplotlib.pyplot as plt

jax.config.update('jax_enable_x64', True)

key = jax.random.PRNGKey(666)

nsamples = 2000

# Samples from a prior
ws = jnp.ones(nsamples) / nsamples
xs = 1. + 0.5 * jax.random.normal(key, shape=(nsamples, 1))

pot_fn = lambda x: jax.scipy.stats.norm.logpdf(-1., x, 0.5)[:, 0]
log_post_ws = pot_fn(xs)
log_post_ws = log_post_ws - jax.scipy.special.logsumexp(log_post_ws)
post_ws = jnp.exp(log_post_ws)

# Resample
key, _ = jax.random.split(key)
ts = jnp.linspace(0, 2, 100)
ys = diffusion_resampling(key, log_post_ws, xs, 0.5, ts, integrator='lord_and_rougemont')

plt.hist(xs[:, 0], bins=64, density=True, alpha=.3, label='Prior')
plt.hist(xs[:, 0], weights=post_ws, bins=64, density=True, color='black', alpha=.3, label='Posterior')
plt.hist(ys[:, 0], weights=post_ws, bins=64, density=True, alpha=.3, label='Resampled')

plt.legend()
plt.tight_layout(pad=.1)
plt.show()

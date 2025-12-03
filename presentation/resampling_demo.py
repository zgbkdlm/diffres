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

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 21})

d = 2

nsamples = 1024

# Generate dummy data
xs = jax.random.normal(key, shape=(nsamples, d))
pot_fn = lambda x: jnp.sum(jax.scipy.stats.norm.logpdf(-0.5, x, 0.5), axis=-1)
log_ws = pot_fn(xs)
log_ws = log_ws - jax.scipy.special.logsumexp(log_ws)

_, samples = ensemble_ot(key, log_ws, xs, eps=0.5)

fig, axes = plt.subplots(ncols=2, sharey='row', sharex=True, figsize=(10, 5))

axes[0].scatter(xs[:, 0], xs[:, 1], edgecolors='none', facecolors='black', s=jnp.exp(log_ws) * nsamples * 4, alpha=.7)
axes[0].set_title('Weighted samples')
axes[1].scatter(samples[:, 0], samples[:, 1], edgecolors='none', facecolors='black', s=4, alpha=.6)
axes[1].set_title('Re-samples')

for ax in axes:
    ax.grid(linestyle='--', alpha=0.3, which='both')
    ax.set_xlim(-2, 1)
    ax.set_ylim(-2, 1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
plt.tight_layout(pad=.1)
plt.savefig('resampling-demo.pdf', transparent=True)
plt.show()

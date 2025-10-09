import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffres.resampling import multinomial, stratified, diffusion_resampling, soft_resampling, gumbel_softmax, ensemble_ot
from diffres.tools import sampling_gm, gm_lin_posterior
from ott.tools.sliced import sliced_wasserstein
from functools import partial

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(666)

c = 5
d = 8
vs = jnp.ones(c) / c
ms = jax.random.uniform(key, minval=-5, maxval=5., shape=(c, d))
key, _ = jax.random.split(key)
_covs = jax.random.normal(key, shape=(c, d))
covs = jnp.einsum('...i,...j->...ij', _covs, _covs) + jnp.eye(d) * 1.
eigvals, eigvecs = jnp.linalg.eigh(covs)

nsamples = 10000

key, _ = jax.random.split(key)
keys = jax.random.split(key, nsamples)
prior_samples = jax.vmap(sampling_gm, in_axes=[0, None, None, None, None])(keys, vs, ms, eigvals, eigvecs)

# True posterior
key, _ = jax.random.split(key)
keys = jax.random.split(key, nsamples)
obs_op = jnp.ones((1, d))
obs_cov = jnp.eye(1)
y_likely = jnp.einsum('ij,kj,k->i', obs_op, ms, vs)

y = y_likely
post_vs, post_ms, post_covs = gm_lin_posterior(y, obs_op, obs_cov, vs, ms, covs)
post_eigvals, post_eigvecs = jnp.linalg.eigh(post_covs)
post_samples = jax.vmap(sampling_gm, in_axes=[0, None, None, None, None])(keys, post_vs, post_ms, post_eigvals,
                                                                          post_eigvecs)

# Importance resampling
@jax.jit
def swd(samples1, samples2, a=None, b=None):
    return sliced_wasserstein(samples1, samples2, a, b, n_proj=1000)[0]


@partial(jax.vmap, in_axes=[0])
def logpdf_likelihood(x):
    return jnp.sum(jax.scipy.stats.norm.logpdf(y, obs_op @ x, obs_cov ** 0.5))


log_ws = logpdf_likelihood(prior_samples)
log_ws = log_ws - jax.scipy.special.logsumexp(log_ws)
ws = jnp.exp(log_ws)

key, _ = jax.random.split(key)

# Multinomial
_, approx_post_multinomial = multinomial(key, log_ws, prior_samples)
print(swd(post_samples, approx_post_multinomial))

plt.scatter(post_samples[:, 0], post_samples[:, 1], s=1, alpha=0.3)
plt.scatter(approx_post_multinomial[:, 0], approx_post_multinomial[:, 1], s=1, alpha=0.3)
plt.show()

# Diffusion resampling
_, approx_post_diffusion = diffusion_resampling(key, log_ws, prior_samples,
                                                -0.5, jnp.linspace(0., 3., 64),
                                                integrator='euler', ode=True)
print(swd(post_samples, approx_post_diffusion))

plt.scatter(post_samples[:, 0], post_samples[:, 1], s=1, alpha=0.3)
plt.scatter(approx_post_diffusion[:, 0], approx_post_diffusion[:, 1], s=1, alpha=0.3)
plt.show()

# Soft
approx_log_post_ws, approx_post_soft = soft_resampling(key, log_ws, prior_samples, alpha=0.8)
print(swd(post_samples, approx_post_soft, b=jnp.exp(approx_log_post_ws)))

plt.scatter(post_samples[:, 0], post_samples[:, 1], s=1, alpha=0.3)
plt.scatter(approx_post_soft[:, 0], approx_post_soft[:, 1], s=1, alpha=0.3)
plt.show()
#
# # Gumbel
# _, approx_post_gumbel = gumbel_softmax(key, log_ws, prior_samples, tau=1e-2)
# print(swd(post_samples, approx_post_gumbel))
#
# plt.scatter(post_samples[:, 0], post_samples[:, 1], s=1, alpha=0.3)
# plt.scatter(approx_post_gumbel[:, 0], approx_post_gumbel[:, 1], s=1, alpha=0.3)
# plt.show()

# # OT
# _, approx_post_ot = ensemble_ot(key, log_ws, prior_samples, eps=0.1)
# print(swd(post_samples, approx_post_ot))
#
# plt.scatter(post_samples[:, 0], post_samples[:, 1], s=1, alpha=0.3)
# plt.scatter(approx_post_ot[:, 0], approx_post_ot[:, 1], s=1, alpha=0.3)
# plt.show()

print(jnp.linalg.det(jnp.cov(post_samples, rowvar=False)))
print(jnp.linalg.det(jnp.cov(approx_post_multinomial, rowvar=False)))
print(jnp.linalg.det(jnp.cov(approx_post_diffusion, rowvar=False)))
print(jnp.linalg.det(jnp.cov(approx_post_soft, rowvar=False)))
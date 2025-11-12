import argparse
import jax
import jax.numpy as jnp
import numpy as np
from diffres.resampling import multinomial, stratified, systematic, diffusion_resampling, soft_resampling, \
    gumbel_softmax, ensemble_ot
from diffres.tools import sampling_gm, gm_lin_posterior
from ott.tools.sliced import sliced_wasserstein
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--id_l', type=int, help='The MC run starting index.')
parser.add_argument('--id_u', type=int, help='The MC run ending index.')
parser.add_argument('--dx', type=int, default=8, help='The x dimension.')
parser.add_argument('--dy', type=int, default=1, help='The y dimension.')
parser.add_argument('--c', type=int, default=5, help='The number of GM components.')
parser.add_argument('--offset', type=float, default=0., help='The offset that makes the observation an outlier.')
parser.add_argument('--nsamples', type=int, default=10_000, help='Number of samples.')
parser.add_argument('--method', type=str, default='multinomial', help='The resampling methods.')
args = parser.parse_args()

jax.config.update("jax_enable_x64", True)
keys_mc = np.load('rnd_keys.npy')[args.id_l:args.id_u + 1]


# Metric
@jax.jit
def swd(samples1, samples2, wx=None, wy=None):
    return sliced_wasserstein(samples1, samples2, wx, wy, n_proj=1000)[0]


sampler_gm = jax.jit(jax.vmap(sampling_gm, in_axes=[0, None, None, None, None, None]))

for mc_id, key_mc in zip(np.arange(args.id_l, args.id_u + 1), keys_mc):
    key, _ = jax.random.split(key_mc)

    # Generate data
    nsamples = args.nsamples
    c = args.c
    d = args.dx
    dy = args.dy
    vs = jnp.ones(c) / c  # GM weights
    ms = jax.random.uniform(key, minval=-5, maxval=5., shape=(c, d))
    key, _ = jax.random.split(key)
    _covs = jax.random.normal(key, shape=(c, d))
    covs = jnp.einsum('...i,...j->...ij', _covs, _covs) + jnp.eye(d) * 1.
    # eigvals, eigvecs = jnp.linalg.eigh(covs)

    key, _ = jax.random.split(key)
    keys = jax.random.split(key, nsamples)
    prior_samples = sampler_gm(keys, vs, ms, _, _, covs)

    # True posterior
    key, _ = jax.random.split(key)
    keys = jax.random.split(key, nsamples)
    obs_op = jnp.ones((dy, d))
    obs_cov = jnp.eye(dy)
    y_likely = jnp.einsum('ij,kj,k->i', obs_op, ms, vs)

    y = y_likely
    post_vs, post_ms, post_covs = gm_lin_posterior(y, obs_op, obs_cov, vs, ms, covs)
    # post_eigvals, post_eigvecs = jnp.linalg.eigh(post_covs)
    post_samples = sampler_gm(keys, post_vs, post_ms, _, _, post_covs)


    # Importance REsampling
    @partial(jax.vmap, in_axes=[0])
    def logpdf_likelihood(x):
        return jnp.sum(jax.scipy.stats.norm.logpdf(y, obs_op @ x, jnp.diag(obs_cov) ** 0.5))


    log_ws = logpdf_likelihood(prior_samples)
    log_ws = log_ws - jax.scipy.special.logsumexp(log_ws)
    ws = jnp.exp(log_ws)

    # Resampling starts
    key, _ = jax.random.split(key)


    @jax.jit
    def resampling(key_, log_ws_, prior_samples_):
        if args.method == 'multinomial':
            lws, xs = multinomial(key_, log_ws_, prior_samples_)
        elif args.method == 'stratified':
            lws, xs = stratified(key_, log_ws_, prior_samples_)
        elif args.method == 'systematic':
            lws, xs = systematic(key_, log_ws_, prior_samples_)
        else:
            raise ValueError(f'Unknown resampling method: {args.method}.')
        return lws, xs


    # Trigger jit and get results
    approx_post_log_ws, approx_post_samples = resampling(key, log_ws, prior_samples)

    # Compute error
    err = swd(post_samples, approx_post_samples)

    # Compute resampling variance (test func = m)
    approx_m = jnp.einsum('n,n...->...', jnp.exp(approx_post_log_ws), approx_post_samples)
    true_m = jnp.einsum('c,c...->...', post_vs, post_ms)
    residual = approx_m - true_m

    # Save result
    print(f'{args.method} (id={mc_id}) has err {err}.')
    np.savez_compressed(f'./gms/results/{args.method}-{mc_id}.npz',
                        post_samples=post_samples, approx_post_log_ws=approx_post_log_ws,
                        approx_post_samples=approx_post_samples, err=err, residual=residual)

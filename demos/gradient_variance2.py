"""
This script illustrates the gradient variance by REINFORCE vs reparametrisation.
This implementation should be statistically the same as `gradient_variance.py`, since the stopped-gradient method
is the same estimator as REINFORCE with smoothing.
"""
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from diffres.resampling import (multinomial, stratified, systematic,
                                diffusion_resampling, multinomial_stopped, ensemble_ot, soft_resampling, gumbel_softmax)
from diffres.feynman_kac import smc_feynman_kac, bootstrap_backward_smoother
from diffres.gaussian_filters import kf as kf_
from diffres.tools import simulate_lgssm, bures, kl
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--nsteps', type=int, default=128, help='Number of time steps.')
parser.add_argument('--nparticles', type=int, default=8, help='Number of nparticles.')
args = parser.parse_args()

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(666)

dx = 1
dy = 1

nsteps = args.nsteps

p1, p2, sig, xi, v0_ = 0.5, 1., 1., 1., 1.

semigroup = p1 * jnp.eye(dx)
trans_cov = sig * jnp.eye(dx)
obs_op = p2 * jnp.ones((dy, dx))
obs_cov = xi * jnp.eye(dy)
m0, v0 = jnp.zeros(dx), v0_ * jnp.eye(dx)


# Filters and loss functions
def kf(params, ys_):
    semigroup_ = params[0] * jnp.eye(dx)
    obs_op_ = params[1] * jnp.ones((dy, dx))
    return kf_(ys_, m0, v0, semigroup_, trans_cov, obs_op_, obs_cov)[:3]


nparticles = args.nparticles
a = -1.
T = 2
dsteps = 128
ts = jnp.linspace(0., T, dsteps + 1)
integrator = 'euler'
ode = True


def diff_resampling(key_, log_ws_, samples_):
    return diffusion_resampling(key_, log_ws_, samples_, a, ts, integrator=integrator, ode=ode, jitter=1e-5)
    # return gumbel_softmax(key_, log_ws_, samples_, 0.1)


def m0_sampler(key_, _):
    rnds = jax.random.normal(key_, shape=(nparticles, dx))
    return m0 + v0_ ** 0.5 * rnds


def pf(key_, params, ys_, resampling_):
    def log_g0(samples, y0):
        return jnp.sum(jax.scipy.stats.norm.logpdf(y0, params[1] * samples, xi ** 0.5), axis=-1)

    def m_log_g(key__, samples, y):
        rnds = jax.random.normal(key__, shape=(nparticles, dx))
        prop_samples = params[0] * samples + sig ** 0.5 * rnds
        log_potentials = jnp.sum(jax.scipy.stats.norm.logpdf(y, params[1] * prop_samples, xi ** 0.5), axis=-1)
        return log_potentials, prop_samples

    return smc_feynman_kac(key_, m0_sampler, log_g0, m_log_g, ys_, nparticles, nsteps,
                           resampling=resampling_, resampling_threshold=1.,
                           return_path=True)


@jax.jit
def diffpf_diff(key_, params, ys_):
    loss_fn = lambda p, y: pf(key_, p, y, diff_resampling)[2]
    return jax.value_and_grad(loss_fn)(params, ys_)


@jax.jit
def diffpf_stopped(key_, params, ys_):
    loss_fn = lambda p, y: pf(key_, p, y, multinomial_stopped)[2]
    return jax.value_and_grad(loss_fn)(params, ys_)


def loss_kf(params, ys_):
    return kf(params, ys_)[-1]


key_simulation, _ = jax.random.split(key)
xs, ys = simulate_lgssm(key_simulation, semigroup, trans_cov, obs_op, obs_cov, m0, v0, nsteps)

params = jnp.array([0.2, 0.2])

loss_true, grad_true = jax.value_and_grad(loss_kf)(params, ys)

nmcs = 1000
keys = jax.random.split(key, num=1000)
losses_dp = np.zeros(nmcs)
losses_rein = np.zeros(nmcs)
grads_dp = np.zeros((nmcs, 2))
grads_rein = np.zeros((nmcs, 2))
for i in range(nmcs):
    print(i)
    loss_dp, grad_dp = diffpf_diff(keys[i], params, ys)
    loss_rein, grad_rein = diffpf_stopped(keys[i], params, ys)
    losses_dp[i] = loss_dp
    losses_rein[i] = loss_rein
    grads_dp[i] = grad_dp
    grads_rein[i] = grad_rein

print(loss_true, grad_true)
print(np.mean(losses_dp), np.mean(grads_dp, axis=0))
print(np.mean(losses_rein), np.mean(grads_rein, axis=0))

print(np.mean((grads_dp - grad_true) ** 2, axis=0) ** 0.5)
print(np.mean((grads_rein - grad_true) ** 2, axis=0) ** 0.5)

print(grads_rein)

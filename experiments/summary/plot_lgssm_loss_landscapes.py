"""
Visually compare the estimated log likelihood functions.
"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from diffres.resampling import (multinomial, stratified, systematic,
                                diffusion_resampling, multinomial_stopped, ensemble_ot, soft_resampling, gumbel_softmax)
from diffres.feynman_kac import smc_feynman_kac
from diffres.gaussian_filters import kf as kf_
from diffres.tools import simulate_lgssm

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 21})

jax.config.update("jax_enable_x64", True)
key_mc = np.load('rnd_keys.npy')[66]

dx = 1
dy = 1

nsteps = 128

p1, p2, sig, xi, v0_ = 0.5, 1., 1., 0.5, 1.

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


nparticles = 32

a = -1.
T = 2.
dsteps = 4
ts = jnp.linspace(0., T, dsteps + 1)
integrator = 'jentzen_and_kloeden'
ode = False
eps = 0.3
tau = 0.2
alpha = 0.2


def resampling_diff(key_, log_ws_, samples_):
    return diffusion_resampling(key_, log_ws_, samples_, a, ts, integrator=integrator, ode=ode)


def resampling_ot(key_, log_ws_, samples_):
    return ensemble_ot(key_, log_ws_, samples_, eps, implicit_diff=False)


def resampling_gumbel(key_, log_ws_, samples_):
    return gumbel_softmax(key_, log_ws_, samples_, tau)


def resampling_soft(key_, log_ws_, samples_):
    return soft_resampling(key_, log_ws_, samples_, alpha)


def resampling_multinomial(key_, log_ws_, samples_):
    return multinomial(key_, log_ws_, samples_)


def m0_sampler(key_, _):
    rnds = jax.random.normal(key_, shape=(nparticles, dx))
    return m0 + v0_ ** 0.5 * rnds


def pf(params, ys_, key_, resampling):
    def log_g0(samples, y0):
        return jnp.sum(jax.scipy.stats.norm.logpdf(y0, params[1] * samples, xi ** 0.5), axis=-1)

    def m_log_g(key__, samples, y):
        rnds = jax.random.normal(key__, shape=(nparticles, dx))
        prop_samples = params[0] * samples + sig ** 0.5 * rnds
        log_potentials = jnp.sum(jax.scipy.stats.norm.logpdf(y, params[1] * prop_samples, xi ** 0.5), axis=-1)
        return log_potentials, prop_samples

    return smc_feynman_kac(key_, m0_sampler, log_g0, m_log_g, ys_, nparticles, nsteps,
                           resampling=resampling, resampling_threshold=1.,
                           return_path=False)[:3]


def loss_fn_kf(params, ys_):
    return kf(params, ys_)[-1]


def loss_fn_diff(params, ys_, key_):
    return pf(params, ys_, key_, resampling_diff)[-1]


def loss_fn_ot(params, ys_, key_):
    return pf(params, ys_, key_, resampling_ot)[-1]


def loss_fn_gumbel(params, ys_, key_):
    return pf(params, ys_, key_, resampling_gumbel)[-1]


def loss_fn_soft(params, ys_, key_):
    return pf(params, ys_, key_, resampling_soft)[-1]


def loss_fn_multinomial(params, ys_, key_):
    return pf(params, ys_, key_, resampling_multinomial)[-1]


vloss_fn_kf = jax.jit(jax.vmap(jax.vmap(loss_fn_kf, in_axes=[0, None]), in_axes=[0, None]))
vloss_fn_diff = jax.jit(jax.vmap(jax.vmap(loss_fn_diff, in_axes=[0, None, None]), in_axes=[0, None, None]))
vloss_fn_ot = jax.jit(jax.vmap(jax.vmap(loss_fn_ot, in_axes=[0, None, None]), in_axes=[0, None, None]))
vloss_fn_gumbel = jax.jit(jax.vmap(jax.vmap(loss_fn_gumbel, in_axes=[0, None, None]), in_axes=[0, None, None]))
vloss_fn_soft = jax.jit(jax.vmap(jax.vmap(loss_fn_soft, in_axes=[0, None, None]), in_axes=[0, None, None]))
vloss_fn_multinomial = jax.jit(
    jax.vmap(jax.vmap(loss_fn_multinomial, in_axes=[0, None, None]), in_axes=[0, None, None]))

ngrids1, ngrids2 = 128, 128
grids_p1 = jnp.linspace(p1 - 0.1, p1 + 0.1, ngrids1)
grids_p2 = jnp.linspace(p2 - 0.1, p2 + 0.1, ngrids2)
mgrids = jnp.meshgrid(grids_p1, grids_p2)
cartesian = jnp.dstack(mgrids)  # (ngrids2, ngrids1, 2)

key_simulation, key_pf = jax.random.split(key_mc)

xs, ys = simulate_lgssm(key_simulation, semigroup, trans_cov, obs_op, obs_cov, m0, v0, nsteps)

losses_kf = vloss_fn_kf(cartesian, ys)  # (ngrids2, ngrids1)
losses_diff = vloss_fn_diff(cartesian, ys, key_pf)
losses_ot = vloss_fn_ot(cartesian, ys, key_pf)
losses_gumbel = vloss_fn_gumbel(cartesian, ys, key_pf)
losses_soft = vloss_fn_soft(cartesian, ys, key_pf)
losses_multinomial = vloss_fn_multinomial(cartesian, ys, key_pf)

# Sharing a same colormap does not work here
vmin = losses_kf.min() - 1
vmax = losses_kf.max() + 1
argmins = [np.unravel_index(np.argmin(a, axis=None), a.shape) for a in [losses_kf, losses_diff, losses_ot, losses_gumbel, losses_soft, losses_multinomial]]

fig, axes = plt.subplots(figsize=(24, 4), ncols=6, sharey='row')

c = axes[0].pcolormesh(*mgrids, losses_kf, cmap=plt.cm.binary, rasterized=True)
fig.colorbar(c)
axes[0].set_title('True loss (KF)')
c = axes[1].pcolormesh(*mgrids, losses_diff, cmap=plt.cm.binary, rasterized=True)
fig.colorbar(c)
axes[1].set_title('Diffusion')
c = axes[2].pcolormesh(*mgrids, losses_ot, cmap=plt.cm.binary, rasterized=True)
fig.colorbar(c)
axes[2].set_title('OT')
c = axes[3].pcolormesh(*mgrids, losses_gumbel, cmap=plt.cm.binary, rasterized=True)
fig.colorbar(c)
axes[3].set_title('Gumbel')
c = axes[4].pcolormesh(*mgrids, losses_soft, cmap=plt.cm.binary, rasterized=True)
fig.colorbar(c)
axes[4].set_title('Soft')
c = axes[5].pcolormesh(*mgrids, losses_multinomial, cmap=plt.cm.binary, rasterized=True)
fig.colorbar(c)
axes[5].set_title('Multinomial')

for argmin, ax in zip(argmins, axes):
    ax.scatter(grids_p1[argmin[1]], grids_p2[argmin[0]], marker='x', s=100, c='tab:red')

for ax in axes:
    ax.scatter(p1, p2, facecolors='none', edgecolors='tab:blue', s=100)
    ax.set_xticks([0.4, 0.5, 0.6])
    ax.set_yticks([0.9, 1.0, 1.1])

axes[0].set_xlabel(r'$\theta_1$')
axes[0].set_ylabel(r'$\theta_2$')

plt.tight_layout(pad=.1)
plt.savefig('lgssm-loss-landscapes.pdf', transparent=True)
plt.show()
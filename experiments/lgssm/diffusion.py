import argparse
import jax
import jax.numpy as jnp
import numpy as np
import jaxopt
from diffres.resampling import (multinomial, stratified, systematic,
                                diffusion_resampling, multinomial_stopped, ensemble_ot, soft_resampling, gumbel_softmax)
from diffres.feynman_kac import smc_feynman_kac
from diffres.gaussian_filters import kf as kf_
from diffres.tools import simulate_lgssm, bures, kl
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--id_l', type=int, help='The MC run starting index.')
parser.add_argument('--id_u', type=int, help='The MC run ending index.')
parser.add_argument('--nsteps', type=int, default=128, help='Number of time steps.')
parser.add_argument('--nparticles', type=int, default=32, help='Number of nparticles.')
parser.add_argument('--a', type=float, default=-0.5, help='The coefficient.')
parser.add_argument('--T', type=float, default=3., help='The terminal time.')
parser.add_argument('--dsteps', type=int, default=8, help='The integration steps of the diffusion.')
parser.add_argument('--integrator', type=str, default='euler', help='The integrator.')
parser.add_argument('--sde', action='store_true', help='The probability flow model or the SDE model.')
args = parser.parse_args()

jax.config.update("jax_enable_x64", True)
keys_mc = np.load('rnd_keys.npy')[args.id_l:args.id_u + 1]

dx = 1
dy = 1

nsteps = args.nsteps

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


nparticles = args.nparticles

a = args.a
T = args.T
dsteps = args.dsteps
ts = jnp.linspace(0., T, dsteps + 1)
integrator = args.integrator
ode = not args.sde


def resampling(key_, log_ws_, samples_):
    return diffusion_resampling(key_, log_ws_, samples_, a, ts, integrator=integrator, ode=ode)


def m0_sampler(key_, _):
    rnds = jax.random.normal(key_, shape=(nparticles, dx))
    return m0 + v0_ ** 0.5 * rnds


def pf(params, ys_, key_, return_path=True):
    def log_g0(samples, y0):
        return jnp.sum(jax.scipy.stats.norm.logpdf(y0, params[1] * samples, xi ** 0.5), axis=-1)

    def m_log_g(key__, samples, y):
        rnds = jax.random.normal(key__, shape=(nparticles, dx))
        prop_samples = params[0] * samples + sig ** 0.5 * rnds
        log_potentials = jnp.sum(jax.scipy.stats.norm.logpdf(y, params[1] * prop_samples, xi ** 0.5), axis=-1)
        return log_potentials, prop_samples

    return smc_feynman_kac(key_, m0_sampler, log_g0, m_log_g, ys_, nparticles, nsteps,
                           resampling=resampling, resampling_threshold=1.,
                           return_path=return_path)[:3]


def loss_fn_kf(params, ys_):
    return kf(params, ys_)[-1]


def loss_fn_pf(params, ys_, key_):
    return pf(params, ys_, key_, return_path=False)[-1]


vloss_fn_kf = jax.jit(jax.vmap(jax.vmap(loss_fn_kf, in_axes=[0, None]), in_axes=[0, None]))
vloss_fn_pf = jax.jit(jax.vmap(jax.vmap(loss_fn_pf, in_axes=[0, None, None]), in_axes=[0, None, None]))
solver = jaxopt.LBFGS(fun=jax.jit(loss_fn_pf), value_and_grad=False, jit=False)

# MC runs
ngrids1, ngrids2 = 128, 128
grids_p1 = jnp.linspace(p1 - 0.1, p1 + 0.1, ngrids1)
grids_p2 = jnp.linspace(p2 - 0.1, p2 + 0.1, ngrids2)
mgrids = jnp.meshgrid(grids_p1, grids_p2)
cartesian = jnp.dstack(mgrids)  # (ngrids2, ngrids1, 2)
jitted_kf = jax.jit(kf)
jitted_pf = jax.jit(pf)

for mc_id, key_mc in zip(np.arange(args.id_l, args.id_u + 1), keys_mc):
    key_simulation, key_pf = jax.random.split(key_mc)

    xs, ys = simulate_lgssm(key_simulation, semigroup, trans_cov, obs_op, obs_cov, m0, v0, nsteps)

    losses_kf = vloss_fn_kf(cartesian, ys)  # (ngrids2, ngrids1)
    losses_pf = vloss_fn_pf(cartesian, ys, key_pf)

    # Compute loss error
    err_loss = jnp.mean((losses_kf - losses_pf) ** 2)

    # Compute filtering error
    mfs_kf, vfs_kf, *_ = jitted_kf(jnp.array([p1, p2]), ys)
    sampless, log_wss, *_ = jitted_pf(jnp.array([p1, p2]), ys, key_pf)
    mfs_pf = jnp.einsum('knd,kn->kd', sampless, jnp.exp(log_wss))
    vfs_pf = jnp.einsum('kni,knj,kn->kij', (sampless - mfs_pf[:, None, :]), (sampless - mfs_pf[:, None, :]),
                        jnp.exp(log_wss))
    err_filtering_kl = jnp.mean(jax.vmap(kl, in_axes=[0, 0, 0, 0])(mfs_kf, vfs_kf, mfs_pf, vfs_pf))
    err_filtering_bures = jnp.mean(jax.vmap(bures, in_axes=[0, 0, 0, 0])(mfs_kf, vfs_kf, mfs_pf, vfs_pf))

    # Optimisation
    init_params = jnp.array([p1 + 1, p2 + 1])
    opt_params, opt_state = solver.run(init_params, ys_=ys, key_=key_pf)

    # Save
    print(f'Diffres | a {a} | T {T} | dsteps={dsteps} | {integrator} {"| ode" if ode else "| sde"} | '
          f'nparticles {nparticles} | id={mc_id} '
          f'| loss err {err_loss} | KL {err_filtering_kl} | Bures {err_filtering_bures} '
          f'| opt params {opt_params}')
    np.savez_compressed(f'./lgssm/results/diffres-{a}-{T}-{dsteps}-{integrator}-{"ode" if ode else "sde"}-'
                        f'{nparticles}-{mc_id}.npz',
                        err_loss=err_loss, err_filtering_kl=err_filtering_kl, err_filtering_bures=err_filtering_bures,
                        opt_params=opt_params,
                        opt_state_iter_num=opt_state.iter_num,
                        opt_state_grad=opt_state.grad,
                        opt_state_error=opt_state.error)

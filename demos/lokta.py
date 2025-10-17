import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from diffres.resampling import (multinomial, stratified, systematic,
                                diffusion_resampling, multinomial_stopped, ensemble_ot, soft_resampling, gumbel_softmax)
from diffres.feynman_kac import smc_feynman_kac
from diffres.nns import NNLoktaVolterra
from diffres.tools import leading_concat
from flax import nnx

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(666)

dx = 2

alp, beta, zeta, gamma, sig = 6., 2., 4., 6., 0.15

t0 = 0.
nsteps = 256
T = 3.
ts = np.linspace(t0, T, nsteps + 1)
dt = T / nsteps


def drift(x):
    return x * (x[..., ::-1] * jnp.array([-beta, zeta]) + jnp.array([alp, -gamma]))


def f(x, dw):
    return x + drift(x) * dt + x * (sig * dw + 0.5 * sig ** 2 * (dw ** 2 - dt))


def g(x):
    # return 2 / (1 + jnp.exp(-5 * jnp.array([x[0], x[0] * x[1]]) + 5))
    return 2 / (1 + jnp.exp(jnp.array([-5 * x[0] + 5, -x[0] * x[1] + 2])))
    # return 2 / (1 + jnp.exp(jnp.array([-5 * x[0] + 5, -5. * jnp.abs(x[0] - x[1]) + 2])))


def logpmf_y_cond_x(y, x):
    return jnp.sum(jax.scipy.stats.poisson.logpmf(y, jax.vmap(g, in_axes=[0])(x)), axis=-1)


# Simulate model
x0 = jnp.array([2., 5.])
key, _ = jax.random.split(key)
y0 = jax.random.poisson(key, g(x0))


def scan_body(carry, elem):
    x = carry
    dw_, key_p = elem

    x = f(x, dw_)
    y = jax.random.poisson(key_p, g(x))
    return x, (x, y)


key, _ = jax.random.split(key)
dws = jax.random.normal(key, shape=(nsteps, dx)) * dt ** 0.5
key, _ = jax.random.split(key)
keys_p = jax.random.split(key, num=nsteps)
_, (xs, ys) = jax.lax.scan(scan_body, x0, (dws, keys_p))
xs = leading_concat(x0, xs)
ys = leading_concat(y0, ys)

# PF
nparticles = 64


def resampling(key_, log_ws_, samples_):
    return diffusion_resampling(key_, log_ws_, samples_, -0.5, jnp.linspace(0., 1., 4),
                                integrator='euler', ode=True)


def m0_sampler(key_, _):
    return jnp.ones((nparticles, dx)) * x0


def log_g0(samples, y0_):
    return logpmf_y_cond_x(y0_, samples)


def m_log_g(key_, samples, y):
    dws_ = jax.random.normal(key_, shape=(nparticles, dx)) * dt ** 0.5
    prop_samples = f(samples, dws_)
    return logpmf_y_cond_x(y, prop_samples), prop_samples


key, _ = jax.random.split(key)
sampless, log_wss, true_nll, *_ = smc_feynman_kac(key, m0_sampler, log_g0, m_log_g, ys, nparticles, nsteps,
                                                  resampling=multinomial, resampling_threshold=1.,
                                                  return_path=True)
print(true_nll)

fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

_ = axes[0].plot(ts, xs[:, 0])
_ = axes[0].plot(ts, xs[:, 1])
for k in range(nsteps + 1):
    _ = axes[0].scatter(ts[k] * np.ones(nparticles), sampless[k, :, 0], s=1, color='tab:blue',
                        alpha=10 * np.exp(log_wss[k]))
    _ = axes[0].scatter(ts[k] * np.ones(nparticles), sampless[k, :, 1], s=1, color='tab:orange',
                        alpha=10 * np.exp(log_wss[k]))
_ = axes[1].scatter(ts, ys[:, 1], s=2)

axes[0].set_xlabel('$t$')
axes[1].set_xlabel('$t$')

for ax in axes:
    ax.grid(linestyle='--', alpha=0.3, which='both')
    ax.legend()

plt.show()

# Learning
key, _ = jax.random.split(key)
model = NNLoktaVolterra(dt=dt, rngs=nnx.Rngs(key))
optimiser = optax.lion(5e-3)
optimiser = nnx.Optimizer(model, optimiser, wrt=nnx.Param)


def loss_fn(model_, key_):
    def m_log_g_(key__, samples, y):
        dws_ = jax.random.normal(key__, shape=(nparticles, dx)) * dt ** 0.5
        prop_samples = model_(samples, dws_)
        return logpmf_y_cond_x(y, prop_samples), prop_samples

    _, _, nll, *_ = smc_feynman_kac(key_, m0_sampler, log_g0, m_log_g_, ys, nparticles, nsteps,
                                    resampling=multinomial, resampling_threshold=1.,
                                    return_path=False)
    return nll


@nnx.jit
def train_step(model_, optimiser_, key_):
    loss_, grads = nnx.value_and_grad(loss_fn)(model_, key_)
    optimiser_.update(model_, grads)
    return loss_


losses = np.zeros(1000)
for i in range(1000):
    key, _ = jax.random.split(key)
    loss = train_step(model, optimiser, key)
    # if (i + 1) % 100 == 0:
    print(f'Step {i} | Loss {loss}')


# Predict
def scan_body(carry, elem):
    x = carry
    dw_, key_p = elem

    x = model(x, dw_)
    y = jax.random.poisson(key_p, g(x))
    return x, (x, y)


key, _ = jax.random.split(key)
dws = jax.random.normal(key, shape=(nsteps, dx)) * dt ** 0.5
key, _ = jax.random.split(key)
keys_p = jax.random.split(key, num=nsteps)
_, (xs, ys) = jax.lax.scan(scan_body, x0, (dws, keys_p))
xs = leading_concat(x0, xs)
ys = leading_concat(y0, ys)
fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

_ = axes[0].plot(ts, xs[:, 0])
_ = axes[0].plot(ts, xs[:, 1])
_ = axes[1].scatter(ts, ys[:, 0], s=2)

axes[0].set_xlabel('$t$')
axes[1].set_xlabel('$t$')

for ax in axes:
    ax.grid(linestyle='--', alpha=0.3, which='both')
    ax.legend()

plt.show()

#
#
#
# def loss_fn_pf(params, ys_, key_):
#     return pf(params, ys_, key_, return_path=False)[-1]
#
#
# vloss_fn_pf = jax.jit(jax.vmap(jax.vmap(loss_fn_pf, in_axes=[0, None, None]), in_axes=[0, None, None]))
# solver = jaxopt.ScipyMinimize(method='L-BFGS-B', fun=loss_fn_pf, jit=True)
#
# # MC runs
# ngrids1, ngrids2 = 128, 128
# grids_p1 = jnp.linspace(p1 - 0.1, p1 + 0.1, ngrids1)
# grids_p2 = jnp.linspace(p2 - 0.1, p2 + 0.1, ngrids2)
# mgrids = jnp.meshgrid(grids_p1, grids_p2)
# cartesian = jnp.dstack(mgrids)  # (ngrids2, ngrids1, 2)
# jitted_pf = jax.jit(pf)
#
# for mc_id, key_mc in zip(np.arange(args.id_l, args.id_u + 1), keys_mc):
#     key_simulation, key_pf = jax.random.split(key_mc)
#
#     xs, ys = simulate_lgssm(key_simulation, semigroup, trans_cov, obs_op, obs_cov, m0, v0, nsteps)
#
#     losses_pf = vloss_fn_pf(cartesian, ys, key_pf)
#
#     # Compute filtering error
#     sampless, log_wss, *_ = jitted_pf(jnp.array([p1, p2]), ys, key_pf)
#     mfs_pf = jnp.einsum('knd,kn->kd', sampless, jnp.exp(log_wss))
#     vfs_pf = jnp.einsum('kni,knj,kn->kij', (sampless - mfs_pf[:, None, :]), (sampless - mfs_pf[:, None, :]),
#                         jnp.exp(log_wss))
#     err_filtering_kl = jnp.mean(jax.vmap(kl, in_axes=[0, 0, 0, 0])(mfs_kf, vfs_kf, mfs_pf, vfs_pf))
#     err_filtering_bures = jnp.mean(jax.vmap(bures, in_axes=[0, 0, 0, 0])(mfs_kf, vfs_kf, mfs_pf, vfs_pf))
#
#     # Optimisation
#     init_params = jnp.array([p1 + 1, p2 + 1])
#     opt_params, opt_state = solver.run(init_params, ys_=ys, key_=key_pf)
#
#     # Save
#     print(f'{args.method} | nparticles {nparticles} | id={mc_id} '
#           f'| loss err {err_loss} | KL {err_filtering_kl} | Bures {err_filtering_bures} '
#           f'| opt params {opt_params}')

import argparse
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from diffres.resampling import (multinomial, stratified, systematic,
                                diffusion_resampling, multinomial_stopped, ensemble_ot, soft_resampling, gumbel_softmax)
from diffres.feynman_kac import smc_feynman_kac
from diffres.nns import NNLoktaVolterra, nnx_save
from diffres.tools import leading_concat

parser = argparse.ArgumentParser()
parser.add_argument('--mc_id', type=int, default=1, help='The MC id.')
parser.add_argument('--nsteps', type=int, default=256, help='Number of time steps.')
parser.add_argument('--mT', type=float, default=3., help='The model end time.')
parser.add_argument('--nparticles', type=int, default=64, help='Number of nparticles.')
parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate.')
parser.add_argument('--niters', type=int, default=1000, help='Number of learning iterations.')
parser.add_argument('--npreds', type=int, default=100, help='Number of ensemble predictions.')
parser.add_argument('--tau', type=float, default=0.5, help='The softmax temperature.')
args = parser.parse_args()

mc_id = args.mc_id
jax.config.update("jax_enable_x64", True)
key = np.load('rnd_keys.npy')[mc_id]

# Model parameters
dx = 2
alp, beta, zeta, gamma, sig = 6., 2., 4., 6., 0.15

t0 = 0.
nsteps = args.nsteps
mT = args.mT
dt = mT / nsteps


def drift(x):
    return x * (x[..., ::-1] * jnp.array([-beta, zeta]) + jnp.array([alp, -gamma]))


def f(x, dw):
    return x + drift(x) * dt + x * (sig * dw + 0.5 * sig ** 2 * (dw ** 2 - dt))


def g(x):
    return 2 / (1 + jnp.exp(jnp.array([-5 * x[0] + 5, -x[0] * x[1] + 2])))


def logpmf_y_cond_x(y, x):
    return jnp.sum(jax.scipy.stats.poisson.logpmf(y, jax.vmap(g, in_axes=[0])(x)), axis=-1)


# Simulate model
x0 = jnp.array([2., 5.])


def simulate_ssm(key_, f_):
    def scan_body(carry, elem):
        x = carry
        dw_, key_p = elem

        x = f_(x, dw_)
        y = jax.random.poisson(key_p, g(x))
        return x, (x, y)

    y0 = jax.random.poisson(key_, g(x0))
    key_, _ = jax.random.split(key_)
    dws = jax.random.normal(key_, shape=(nsteps, dx)) * dt ** 0.5
    key_, _ = jax.random.split(key_)
    keys_p = jax.random.split(key_, num=nsteps)
    _, (xs_, ys_) = jax.lax.scan(scan_body, x0, (dws, keys_p))
    return leading_concat(x0, xs_), leading_concat(y0, ys_)


key, _ = jax.random.split(key)
xs, ys = simulate_ssm(key, f)

# Run PF with the true model
nparticles = args.nparticles
tau = args.tau


@jax.jit
def resampling(key_, log_ws_, samples_):
    return gumbel_softmax(key_, log_ws_, samples_, tau)


def m0_sampler(key_, _):
    return jnp.ones((nparticles, dx)) * x0


def log_g0(samples, y0_):
    return logpmf_y_cond_x(y0_, samples)


def m_log_g(key_, samples, y):
    dws_ = jax.random.normal(key_, shape=(nparticles, dx)) * dt ** 0.5
    prop_samples = f(samples, dws_)
    return logpmf_y_cond_x(y, prop_samples), prop_samples


key, _ = jax.random.split(key)
sampless, log_wss, target_nll, *_ = smc_feynman_kac(key, m0_sampler, log_g0, m_log_g, ys, nparticles, nsteps,
                                                    resampling=multinomial, resampling_threshold=1.,
                                                    return_path=True)

# NN learning
key, _ = jax.random.split(key)
model = NNLoktaVolterra(dt=dt, rngs=nnx.Rngs(key))
optimiser = optax.lion(args.lr)
optimiser = nnx.Optimizer(model, optimiser, wrt=nnx.Param)


def loss_fn(model_, key_):
    def m_log_g_(key__, samples, y):
        dws_ = jax.random.normal(key__, shape=(nparticles, dx)) * dt ** 0.5
        prop_samples = model_(samples, dws_)
        return logpmf_y_cond_x(y, prop_samples), prop_samples

    _, _, nll, *_ = smc_feynman_kac(key_, m0_sampler, log_g0, m_log_g_, ys, nparticles, nsteps,
                                    resampling=resampling, resampling_threshold=1.,
                                    return_path=False)
    return nll


@nnx.jit
def train_step(model_, optimiser_, key_):
    loss_, grads = nnx.value_and_grad(loss_fn)(model_, key_)
    optimiser_.update(model_, grads)
    return loss_


print_prefix = f'Gumbel ({mc_id}) | tau {tau}'
filename_prefix = f'gumbel-{tau}-'
losses = np.zeros(args.niters)
for i in range(args.niters):
    key, _ = jax.random.split(key)
    loss = train_step(model, optimiser, key)
    losses[i] = loss
    print(print_prefix + f' | Iter {i} | loss {loss} | target loss {target_nll}')
nnx_save(model, './lokta/checkpoints/' + filename_prefix + f'{mc_id}')


# Predict and compute error (this can do offline)
def pred_err_per_path(key_):
    xs_pred, _ = simulate_ssm(key_, model)
    return jnp.mean((xs_pred - xs) ** 2) ** 0.5


key, _ = jax.random.split(key)
keys = jax.random.split(key, num=args.npreds)
pred_err = jnp.mean(jax.vmap(pred_err_per_path)(keys))
print(print_prefix + f' | Prediction RMSE {pred_err}')
np.savez_compressed('./lokta/results/' + filename_prefix + f'{mc_id}.npz',
                    losses=losses, pred_err=pred_err)

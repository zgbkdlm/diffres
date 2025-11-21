"""
Visualise some predictions from the Lokta model in appendix.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx
from diffres.nns import NNLoktaVolterra, nnx_load
from diffres.tools import leading_concat

jax.config.update("jax_enable_x64", True)

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 21})

# Global params
nparticles = 64
num_mcs = 20
niters = 1000
training_iters = np.arange(niters)
npreds = 100


def check_nan(loss, pred_err):
    if np.isnan(loss) or np.isnan(pred_err) or pred_err > 1000.:
        return True
    else:
        return False


diff_methods = ['-1.0-1.0-4-euler-sde',
                '-1.0-2.0-16-euler-sde']
diff_labels = ['$T=1$\n $K=4$',
               '$T=2$\n $K=16$']
epss = [0.3, 0.5]
taus = [0.3, ]
alphas = [0.7, ]

# Model parameters
dx = 2
alp, beta, zeta, gamma, sig = 6., 2., 4., 6., 0.15

t0 = 0.
nsteps = 256
mT = 3.
dt = mT / nsteps
model_ts = np.linspace(t0, mT, nsteps + 1)


def drift(x):
    return x * (x[..., ::-1] * jnp.array([-beta, zeta]) + jnp.array([alp, -gamma]))


def f(x, dw):
    return x + drift(x) * dt + x * (sig * dw + 0.5 * sig ** 2 * (dw ** 2 - dt))


def g(x):
    return 5 / (1 + jnp.exp(jnp.array([-5 * x[0] + 4, -x[0] * x[1] + 4])))


def logpmf_y_cond_x(y, x):
    return jnp.sum(jax.scipy.stats.poisson.logpmf(y, jax.vmap(g, in_axes=[0])(x)), axis=-1)


# Simulate model
x0 = jnp.array([2., 5.])
model = NNLoktaVolterra(dt=dt, rngs=nnx.Rngs(666))


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


mc_id = 7
mc_key = np.load('rnd_keys.npy')[mc_id]  # This precisely reproduces the original seed
key, _ = jax.random.split(mc_key)
true_path, _ = simulate_ssm(key, f)
keys = jax.random.split(key, num=npreds)
true_paths, _ = jax.vmap(simulate_ssm, in_axes=[0, None])(keys, f)
mc_mean = jnp.mean(true_paths, axis=0)
mc_var = jnp.var(true_paths, axis=0)
key, _ = jax.random.split(key)

# Diffusion
for method in diff_methods:
    filename_prefix = f'./lokta/results/diffres-' + method + '-'
    data = np.load(filename_prefix + f'{mc_id}.npz')
    model = nnx_load(model, './lokta/checkpoints/diffres-' + method + f'-{mc_id}', display=True)

    keys = jax.random.split(key, num=npreds)
    pred_paths, _ = jax.vmap(simulate_ssm, in_axes=[0, None])(keys, model)

    fig, ax = plt.subplots()

    ax.plot(model_ts, true_path[:, 1], linewidth=3, c='black', label='True path')
    ax.fill_between(model_ts,
                    mc_mean[:, 1] - 1.96 * jnp.sqrt(mc_var[:, 1]),
                    mc_mean[:, 1] + 1.96 * jnp.sqrt(mc_var[:, 1]),
                    color='black',
                    edgecolor='none',
                    alpha=0.15, label='0.95 confidence')
    for j in range(10):
        ax.plot(model_ts, pred_paths[j, :, 1],
                linestyle='--', c='black', alpha=.3, label='Predicted paths' if j == 9 else '')
    ax.grid(linestyle='--', alpha=0.3, which='both')
    ax.set_xlabel('Time $t$ (s)')
    ax.set_ylabel('$R(t)$')
    plt.legend()
    plt.tight_layout(pad=.1)
    plt.show()

# OT
for eps in epss:
    filename_prefix = f'./lokta/results/ot-{eps}-'
    data = np.load(filename_prefix + f'{mc_id}.npz')
    model = nnx_load(model, f'./lokta/checkpoints/ot-{eps}-{mc_id}', display=True)

    keys = jax.random.split(key, num=npreds)
    pred_paths, _ = jax.vmap(simulate_ssm, in_axes=[0, None])(keys, model)

    fig, ax = plt.subplots()

    ax.plot(model_ts, true_path[:, 1], linewidth=3, c='black', label='True path')
    ax.fill_between(model_ts,
                    mc_mean[:, 1] - 1.96 * jnp.sqrt(mc_var[:, 1]),
                    mc_mean[:, 1] + 1.96 * jnp.sqrt(mc_var[:, 1]),
                    color='black',
                    edgecolor='none',
                    alpha=0.15, label='0.95 confidence')
    for j in range(10):
        ax.plot(model_ts, pred_paths[j, :, 1],
                linestyle='--', c='black', alpha=.3, label='Predicted paths' if j == 9 else '')
    ax.grid(linestyle='--', alpha=0.3, which='both')
    ax.set_xlabel('Time $t$ (s)')
    ax.set_ylabel('$R(t)$')
    plt.legend()
    plt.tight_layout(pad=.1)
    plt.show()

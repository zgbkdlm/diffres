"""
In this notebook we demonstrate the distinct features between OT and diffres

Definition of "fully-differentiable": the random variable is differentiable.

OT (entropy): Yes,
OT (Kantorovich): No
SDE (cont.): No
SDE (disc): Yes

SDE-cont: reinterpretation
SDE-disc: relaxation.

epsilon then corresponds to the discretisation step not T.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffres.integators import euler_maruyama, jentzen_and_kloeden, tweedie
from diffres.tools import sampling_gm, gm_lin_posterior
from diffres.resampling import ensemble_ot, diffusion_resampling, gumbel_softmax, soft_resampling
from functools import partial

jax.config.update('jax_enable_x64', True)

key = jax.random.PRNGKey(666)

# Generate data
# Three particles move along distinct orbits
ndata = 3
thetas = jnp.linspace(-1, 1., 100)


def fn_payoff(x):
    return jnp.prod(x, axis=-1)


def data_generator(theta):
    data_ = jnp.concatenate([jnp.array([[jnp.sin(2 * jnp.pi * theta), jnp.cos(2 * jnp.pi * theta)]]),
                             jnp.array([[-jnp.sin(2 * jnp.pi * theta), -jnp.cos(2 * jnp.pi * theta)]]) * 0.6,
                             jnp.array([[jnp.cos(2 * jnp.pi * theta), jnp.sin(2 * jnp.pi * theta)]]) * 0.3],
                            axis=0)
    ws_ = jnp.abs(jnp.array([jnp.cos(2 * jnp.pi * theta),
                             jnp.sin(2 * jnp.pi * theta),
                             jnp.cos(2 * jnp.pi * (theta - 0.5))]))
    ws_ = ws_ / ws_.sum()
    log_ws_ = jnp.log(ws_)
    return data_, log_ws_, jnp.sum(ws_ * fn_payoff(data_))

datas, log_wss, payoffs = jax.vmap(data_generator)(thetas)

# Define two inspectors
ts = jnp.linspace(0., 1.5, 10000)


def inspector_diffres(key_, theta):
    data_, log_ws_, payoff = data_generator(theta)
    _, re_samples = diffusion_resampling(key_, log_ws_, data_, -1., ts, 'tweedie', ode=False)
    re_payoff = jnp.mean(fn_payoff(re_samples))
    return re_samples, re_payoff


def inspector_ot(key_, theta):
    data_, log_ws_, payoff = data_generator(theta)
    _, re_samples = ensemble_ot(key_, log_ws_, data_, eps=0.1)
    re_payoff = jnp.mean(fn_payoff(re_samples))
    return re_samples, re_payoff


def inspector_gumbel(key_, theta):
    data_, log_ws_, payoff = data_generator(theta)
    _, re_samples = gumbel_softmax(key_, log_ws_, data_, tau=0.01)
    re_payoff = jnp.mean(fn_payoff(re_samples))
    return re_samples, re_payoff


resamples_diffres, re_payoffs_diffres = jax.vmap(inspector_diffres, in_axes=[None, 0])(key, thetas)

fig, axes = plt.subplots(ncols=5, figsize=(18, 4), sharey=True, sharex=True)
for i in range(3):
    axes[0].scatter(datas[:, i, 0], datas[:, i, 1], s=jnp.exp(log_wss[:, i]) * 50, label='data', alpha=.5)

for n in range(3):
    axes[n + 1].scatter(resamples_diffres[:, n, 0], resamples_diffres[:, n, 1], s=1, label='resample', alpha=.5)

axes[-1].plot(thetas, payoffs)
axes[-1].plot(thetas, re_payoffs_diffres)
plt.show()

# plt.scatter(post_samples[:, 0], post_samples[:, 1], s=1, label='post', alpha=.5)
# plt.scatter(prior_samples[:, 0], prior_samples[:, 1], s=1, label='prior', alpha=.5)
# plt.scatter(resamples_[:, 0], resamples_[:, 1], s=1, label='re', alpha=.5)
# plt.legend()
# plt.show()
#
# for i in range(10):
#     plt.plot(paths[:, i, 0], linewidth=1, alpha=.5)
# plt.show()

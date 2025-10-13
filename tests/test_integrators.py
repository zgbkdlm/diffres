import pytest
import jax
import jax.numpy as jnp
import numpy.testing as npt
from ott.tools.sliced import sliced_wasserstein
from diffres.tools import make_gm_bridge, sampling_gm
from diffres.integators import euler_maruyama, lord_and_rougemont, jentzen_and_kloeden, tweedie
from functools import partial

jax.config.update("jax_enable_x64", True)

key = jax.random.PRNGKey(666)

# GM
c = 5
d = 2
ws = jnp.ones(c) / c
ms = -5. + 10 * jax.random.uniform(key, shape=(c, d))
key, _ = jax.random.split(key)
_vs = jax.random.normal(key, shape=(c, d))
covs = jnp.einsum('...i,...j->...ij', _vs, _vs) + jnp.eye(d) * 0.1
eigvals, eigvecs = jnp.linalg.eigh(covs)

# Fwd SDE
nsamples = 10000
a = -1.
b = 1.

# Times
t0 = 0.
T = 2.
nsteps = 16
ts = jnp.linspace(t0, T, nsteps + 1)

# Generate data
wTs, mTs, eigvalTs, score, _, _ = make_gm_bridge(ws, ms, eigvals, eigvecs, a, b, t0, T,
                                                 fwd_denoising=False)


# Reversal SDE (in fwd time flow)
def f(x, t):
    return b ** 2 * score(x, T - t)


def drift(x, t):
    return -a * x + f(x, t)


key, _ = jax.random.split(key)
keys = jax.random.split(key, nsamples)
x0s = jax.vmap(sampling_gm, in_axes=[0, None, None, None, None])(keys, ws, ms, eigvals, eigvecs)

key, _ = jax.random.split(key)
keys = jax.random.split(key, nsamples)
xTs = jax.vmap(sampling_gm, in_axes=[0, None, None, None, None])(keys, wTs, mTs, eigvalTs, eigvecs)


@jax.jit
def swd(samples1, samples2):
    return sliced_wasserstein(samples1, samples2, n_proj=1000)[0]


def body_euler(carry, elem):
    x = carry
    t_km1, tk, rnd = elem

    dt = tk - t_km1
    m_, scale_ = euler_maruyama(drift, b, x, t_km1, dt)
    return m_ + scale_ * rnd, None


def body_lord(carry, elem):
    x = carry
    t_km1, tk, rnd = elem

    dt = tk - t_km1
    m_, scale_ = lord_and_rougemont(-a, f, b, x, t_km1, dt)
    return m_ + scale_ * rnd, None


def body_jentzen(carry, elem):
    x = carry
    t_km1, tk, rnd = elem

    dt = tk - t_km1
    m_, scale_ = jentzen_and_kloeden(-a, f, b, x, t_km1, dt)
    return m_ + scale_ * rnd, None


def body_tweedie(carry, elem):
    x = carry
    t_km1, tk, rnd = elem

    dt = tk - t_km1
    sg = jnp.exp(a * dt)
    trans_var = b ** 2 / (2 * a) * (jnp.exp(2 * a * dt) - 1)
    m_, scale_ = tweedie(sg, trans_var, score, 0., x, T - t_km1)
    return m_ + scale_ * rnd, None


@jax.jit
@partial(jax.vmap, in_axes=[0, 0])
def sampler_euler(key_, x_ref):
    rnds = jax.random.normal(key_, shape=(nsteps, d))
    return jax.lax.scan(body_euler, x_ref, (ts[:-1], ts[1:], rnds))[0]


@jax.jit
@partial(jax.vmap, in_axes=[0, 0])
def sampler_lord(key_, x_ref):
    rnds = jax.random.normal(key_, shape=(nsteps, d))
    return jax.lax.scan(body_lord, x_ref, (ts[:-1], ts[1:], rnds))[0]


@jax.jit
@partial(jax.vmap, in_axes=[0, 0])
def sampler_jentzen(key_, x_ref):
    rnds = jax.random.normal(key_, shape=(nsteps, d))
    return jax.lax.scan(body_jentzen, x_ref, (ts[:-1], ts[1:], rnds))[0]


@jax.jit
@partial(jax.vmap, in_axes=[0, 0])
def sampler_tweedie(key_, x_ref):
    rnds = jax.random.normal(key_, shape=(nsteps, d))
    return jax.lax.scan(body_tweedie, x_ref, (ts[:-1], ts[1:], rnds))[0]


def test_integrators():
    key = jax.random.PRNGKey(999)
    keys = jax.random.split(key, num=nsamples)
    x0s_euler = sampler_euler(keys, xTs)
    x0s_lord = sampler_lord(keys, xTs)
    x0s_jentzen = sampler_jentzen(keys, xTs)
    x0s_tweedie = sampler_tweedie(keys, xTs)

    err_euler = swd(x0s, x0s_euler)
    err_lord = swd(x0s, x0s_lord)
    err_jentzen = swd(x0s, x0s_jentzen)
    err_tweedie = swd(x0s, x0s_tweedie)

    npt.assert_allclose(err_euler, 0., atol=3e-2)
    npt.assert_allclose(err_lord, 0., atol=2e-2)
    npt.assert_allclose(err_jentzen, 0., atol=2e-2)
    npt.assert_allclose(err_tweedie, 0., atol=3e-2)

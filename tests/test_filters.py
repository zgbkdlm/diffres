import pytest
import jax
import jax.numpy as jnp
import numpy.testing as npt
from diffres.resampling import (multinomial, stratified, systematic,
                                diffusion_resampling, multinomial_stopped, ensemble_ot)
from diffres.feynman_kac import smc_feynman_kac
from diffres.tools import logpdf_mvn_chol, chol_solve, leading_concat

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(666)

d = 2
t0 = 0.
nsteps = 31
T = 1.
ts = jnp.linspace(t0, T, nsteps + 1)
dt = T / nsteps

m0 = jnp.zeros(d)
v0 = jnp.eye(d)

A = jnp.array([[1, dt],
               [0, 1]])
Sigma = jnp.array([[dt ** 3 / 3, dt ** 2 / 2],
                   [dt ** 2 / 2, dt]])
chol = jnp.linalg.cholesky(Sigma)
H = jnp.eye(d)
xi = 0.1


def body(carry, elem):
    x = carry
    rnd = elem

    x = A @ x + chol @ rnd[:d]
    y = H @ x + xi ** 0.5 * rnd[d:]
    return x, (x, y)


x0 = m0 + jnp.linalg.cholesky(v0) @ jax.random.normal(key, shape=(d,))
key, _ = jax.random.split(key)
y0 = H @ x0 + xi ** 0.5 * jax.random.normal(key, shape=(d,))

key, _ = jax.random.split(key)
rnds = jax.random.normal(key, shape=(nsteps, 2 * d))
_, (xs, ys) = jax.lax.scan(body, x0, rnds)

ys = leading_concat(y0, ys)


def kf_update(mp, vp, y):
    S = H @ vp @ H.T + xi
    ch = jax.scipy.linalg.cho_factor(S)
    C = vp @ H.T
    mf = mp + C @ jax.scipy.linalg.cho_solve(ch, y - H @ mp)
    vf = vp - C @ jax.scipy.linalg.cho_solve(ch, C.T)
    nll = -jax.scipy.stats.multivariate_normal.logpdf(y, H @ mp, S)
    return mf, vf, nll


def kf_body(carry, elem):
    mf, vf = carry
    y = elem

    mp = A @ mf
    vp = A @ vf @ A.T + Sigma

    mf, vf, nll = kf_update(mp, vp, y)
    return (mf, vf), (mf, vf, nll)


mf0, vf0, nll0 = kf_update(m0, v0, ys[0])
_, (mfs, vfs, nlls) = jax.lax.scan(kf_body, (mf0, vf0), ys[1:])
true_nll = nll0 + jnp.sum(nlls)

nparticles = 1000


def m0_sampler(key_):
    rnds = jax.random.normal(key_, shape=(nparticles, d))
    return m0 + rnds @ jnp.linalg.cholesky(v0).T


def log_g0(samples):
    return jnp.sum(jax.scipy.stats.norm.logpdf(ys[0], samples @ H.T, xi), axis=-1)


def m_log_g(key_, samples, pytree):
    y = pytree
    rnds = jax.random.normal(key_, shape=(nparticles, d))
    prop_samples = samples @ A.T + rnds @ chol.T
    log_potentials = jnp.sum(jax.scipy.stats.norm.logpdf(y, prop_samples @ H.T, xi), axis=-1)
    return log_potentials, prop_samples


@pytest.mark.parametrize('r', [multinomial, multinomial_stopped, stratified, systematic])
def test_smc(r):
    _, _, nll_pf, *_ = smc_feynman_kac(key, m0_sampler, log_g0, m_log_g, ys[1:], nparticles, nsteps,
                                       resampling=r, resampling_threshold=1.,
                                       return_path=True)

    npt.assert_allclose(nll_pf, true_nll, rtol=1e-1)


@pytest.mark.parametrize('integrator', ['euler', 'ode', 'lord_and_rougemont', 'jentzen_and_kloeden'])
def test_diffres(integrator):
    def r(key_, log_ws, samples):
        return diffusion_resampling(key_, log_ws, samples, -0.5, jnp.linspace(0., 1., 16), integrator=integrator)

    _, _, nll_pf, *_ = smc_feynman_kac(key, m0_sampler, log_g0, m_log_g, ys[1:], nparticles, nsteps,
                                       resampling=r, resampling_threshold=1.,
                                       return_path=True)
    npt.assert_allclose(nll_pf, true_nll, rtol=1e-1)


def test_ensemble_ot():
    def r(key_, log_ws, samples):
        return ensemble_ot(key_, log_ws, samples, eps=0.1)

    _, _, nll_pf, *_ = smc_feynman_kac(key, m0_sampler, log_g0, m_log_g, ys[1:], nparticles, nsteps,
                                       resampling=r, resampling_threshold=1.,
                                       return_path=True)
    npt.assert_allclose(nll_pf, true_nll, rtol=1e-1)

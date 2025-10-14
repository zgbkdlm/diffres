import pytest
import jax
import jax.numpy as jnp
import numpy.testing as npt
from diffres.resampling import (multinomial, stratified, systematic,
                                diffusion_resampling, multinomial_stopped, ensemble_ot)
from diffres.feynman_kac import smc_feynman_kac
from diffres.gaussian_filters import kf, rts

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(666)

ell = 1.
sigma = 1.

dx = 1
dy = 1
obs_op = jnp.ones((dy, dx))
xi = 1.
obs_cov = jnp.eye(dy) * xi

T = 2.
nsteps = 100
ts = jnp.linspace(0., 2., nsteps + 1)
dt = T / nsteps


def cov_fn(t, s):
    return sigma ** 2 * jnp.exp(-jnp.abs(t[:, None] - s[None, :]) / ell)


ys = jax.random.normal(key, shape=(nsteps + 1,))

# GP regression
cov = cov_fn(ts, ts)
G = cov + jnp.eye(nsteps + 1) * xi
chol = jax.scipy.linalg.cho_factor(G)
gp_mean = cov @ jax.scipy.linalg.cho_solve(chol, ys)
gp_cov = cov - cov @ jax.scipy.linalg.cho_solve(chol, cov)
gp_nll = -jax.scipy.stats.multivariate_normal.logpdf(ys, jnp.zeros(nsteps + 1), G)

# KF
m0, v0 = jnp.zeros(dx), sigma ** 2 * jnp.eye(dx)
semigroup = jnp.eye(dx) * jnp.exp(-1 / ell * dt)
trans_cov = sigma ** 2 * (1 - jnp.exp(-2 / ell * dt)) * jnp.eye(dx)
mfs, vfs, nll, mps, vps = kf(ys, m0, v0, semigroup, trans_cov, obs_op, obs_cov)
mss, vss = rts(mfs, vfs, mps, vps, semigroup)


def test_kf():
    npt.assert_allclose(mss[:, 0], gp_mean)
    npt.assert_allclose(vss[:, 0, 0], jnp.diag(gp_cov))
    npt.assert_allclose(nll, gp_nll)


nparticles = 1000


def m0_sampler(key_, _):
    rnds = jax.random.normal(key_, shape=(nparticles, dx))
    return m0 + rnds @ jnp.linalg.cholesky(v0).T


def log_g0(samples, y0):
    return jnp.sum(jax.scipy.stats.norm.logpdf(y0, samples @ obs_op.T, xi ** 0.5), axis=-1)


def m_log_g(key_, samples, pytree):
    y = pytree
    rnds = jax.random.normal(key_, shape=(nparticles, dx))
    prop_samples = samples @ semigroup.T + rnds @ (trans_cov ** 0.5).T
    log_potentials = jnp.sum(jax.scipy.stats.norm.logpdf(y, prop_samples @ obs_op.T, xi ** 0.5), axis=-1)
    return log_potentials, prop_samples


@pytest.mark.parametrize('r', [multinomial, multinomial_stopped, stratified, systematic])
def test_smc(r):
    _, _, nll_pf, *_ = smc_feynman_kac(key, m0_sampler, log_g0, m_log_g, ys, nparticles, nsteps,
                                       resampling=r, resampling_threshold=1.,
                                       return_path=True)

    npt.assert_allclose(nll_pf, nll, rtol=6e-2)


@pytest.mark.parametrize('integrator', ['euler', 'lord_and_rougemont', 'jentzen_and_kloeden', 'tweedie'])
def test_diffres(integrator):
    def r(key_, log_ws, samples):
        return diffusion_resampling(key_, log_ws, samples, -0.5, jnp.linspace(0., 2., 8),
                                    integrator=integrator, ode=False)

    _, _, nll_pf, *_ = smc_feynman_kac(key, m0_sampler, log_g0, m_log_g, ys, nparticles, nsteps,
                                       resampling=r, resampling_threshold=1.,
                                       return_path=True)
    npt.assert_allclose(nll_pf, nll, rtol=5e-2)


def test_ensemble_ot():
    def r(key_, log_ws, samples):
        return ensemble_ot(key_, log_ws, samples, eps=0.1)

    _, _, nll_pf, *_ = smc_feynman_kac(key, m0_sampler, log_g0, m_log_g, ys, nparticles, nsteps,
                                       resampling=r, resampling_threshold=1.,
                                       return_path=True)
    npt.assert_allclose(nll_pf, nll, rtol=7e-2)

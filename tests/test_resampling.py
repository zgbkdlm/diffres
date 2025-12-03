import pytest
import jax
import jax.numpy as jnp
import numpy.testing as npt
from diffres.resampling import (multinomial, stratified, systematic,
                                diffusion_resampling, diffusion_resampling_generic, multinomial_stopped,
                                ensemble_ot, soft_resampling, gumbel_softmax)
from ott.tools.sliced import sliced_wasserstein

jax.config.update("jax_enable_x64", True)


@jax.jit
def swd(samples1, samples2, a, b):
    return sliced_wasserstein(samples1, samples2, a, b, n_proj=1000)[0]


key = jax.random.PRNGKey(666)
nsamples = 10000
xs = jax.random.uniform(key, minval=-2., maxval=2., shape=(nsamples, 1))
log_g = lambda x: jnp.sum(jax.scipy.stats.norm.logpdf(0., x, 1.), axis=-1)
log_ws_ = log_g(xs)
log_ws = log_ws_ - jax.scipy.special.logsumexp(log_ws_)

key_resampling, _ = jax.random.split(key)

soft_r = lambda k, lw, x: soft_resampling(k, lw, x, alpha=1 - 1e-3)
gumbel_r = lambda k, lw, x: gumbel_softmax(k, lw, x, tau=1e-3)


@pytest.mark.parametrize('r', [soft_r, gumbel_r])
def test_misc_resamplings(r):
    resampled_log_ws, resampled_xs = r(key_resampling, log_ws, xs)
    npt.assert_allclose(swd(resampled_xs, xs, jnp.exp(resampled_log_ws), jnp.exp(log_ws)), 0., atol=1e-4)


@pytest.mark.parametrize('integrator', ['euler', 'lord_and_rougemont', 'jentzen_and_kloeden', 'diffrax', 'tweedie'])
@pytest.mark.parametrize('ode', [True, False])
def test_diffres(integrator, ode):
    if integrator == 'diffrax' and not ode:
        pytest.skip('pass')
    if integrator == 'tweedie' and ode:
        pytest.skip('pass')
    ts = jnp.linspace(0., 1., 16)
    resampled_log_ws, resampled_xs = diffusion_resampling(key_resampling, log_ws, xs, -0.5, ts,
                                                          integrator=integrator, ode=ode)
    npt.assert_allclose(swd(resampled_xs, xs, jnp.exp(resampled_log_ws), jnp.exp(log_ws)), 0., atol=1e-3)


def test_diffres_versions():
    """The two correspond to exactly the same implementation.
    """
    a = -1.
    ts = jnp.linspace(0., 1., 16)
    resampled_log_ws, resampled_xs = diffusion_resampling(key_resampling, log_ws, xs, a, ts,
                                                          integrator='euler', ode=False)

    # Test the generic resampler
    ws = jnp.exp(log_ws)
    mu = jnp.einsum('i,i...->...', ws, xs)
    stat_vars = jnp.einsum('i,i...->...', ws, (xs - mu) ** 2)
    b2 = -a * stat_vars

    def ref_sampler(key_):
        return mu + jax.random.normal(key_, shape=(nsamples, 1)) * stat_vars ** 0.5

    def score_ref(x):
        return -(x - mu) / stat_vars

    def logpdf_trans(xt, xs_, t, s):
        delta = t - s
        semigroup = jnp.exp(a * delta)
        sigts = stat_vars * (1 - jnp.exp(2 * a * delta))
        return jnp.sum(jax.scipy.stats.norm.logpdf(xt, xs_ * semigroup + mu * (1 - semigroup), sigts ** 0.5), axis=-1)

    resampled_log_ws2, resampled_xs2 = diffusion_resampling_generic(key_resampling, log_ws, xs,
                                                                    ref_sampler, logpdf_trans, score_ref, b2 ** 0.5, ts,
                                                                    ode=False)
    npt.assert_allclose(resampled_log_ws, resampled_log_ws2)
    npt.assert_allclose(resampled_xs, resampled_xs2)


def test_ensemble_ot():
    resampled_log_ws, resampled_xs = ensemble_ot(_, log_ws, xs, eps=0.1)
    npt.assert_allclose(swd(resampled_xs, xs, jnp.exp(resampled_log_ws), jnp.exp(log_ws)), 0., atol=3e-3)

import pytest
import jax
import jax.numpy as jnp
import numpy.testing as npt
from diffres.resampling import multinomial, stratified, systematic, diffusion_resampling
from ott.tools.sliced import sliced_wasserstein

jax.config.update("jax_enable_x64", True)


@jax.jit
def swd(samples1, samples2):
    return sliced_wasserstein(samples1, samples2, n_proj=1000)[0]


key = jax.random.PRNGKey(666)
xs = jax.random.uniform(key, minval=-2., maxval=2., shape=(10000, 1))
log_g = lambda x: jnp.sum(jax.scipy.stats.norm.logpdf(0., x, 1.), axis=-1)
log_ws_ = log_g(xs)
log_ws = log_ws_ - jax.scipy.special.logsumexp(log_ws_)
ws = jnp.exp(log_ws)

key_resampling, _ = jax.random.split(key)


@pytest.mark.parametrize('r', [multinomial, stratified, systematic])
def test_misc_resamplings(r):
    resampled_xs = xs[r(key_resampling, ws)]
    npt.assert_allclose(swd(resampled_xs, xs), 0., atol=1e-1)


@pytest.mark.parametrize('integrator', ['euler', 'lord_and_rougemont', 'jentzen_and_kloeden'])
def test_diffres(integrator):
    ts = jnp.linspace(0., 2., 10)
    resampled_xs = diffusion_resampling(key_resampling, log_ws, xs, -0.5, ts, integrator=integrator)
    npt.assert_allclose(swd(resampled_xs, xs), 0., atol=1e-1)

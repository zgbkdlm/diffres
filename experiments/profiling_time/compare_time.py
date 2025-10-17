import argparse
import timeit
import jax
import jax.numpy as jnp
import numpy as np
from diffres.resampling import multinomial, stratified, systematic, diffusion_resampling, soft_resampling, \
    gumbel_softmax, ensemble_ot

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(666)

d = 8

ntries = 10  # this is sufficient, as the running times (of different methods) are fairly different
nsampless = [128, 256, 512, 1024, 2048, 4096, 8192]

times_multinomial = np.zeros(ntries)
times_soft = np.zeros(ntries)
times_gumbel = np.zeros(ntries)

for nsamples in nsampless:
    # Generate dummy data
    xs = jax.random.normal(key, shape=(nsamples, d))
    pot_fn = lambda x: jnp.sum(jax.scipy.stats.norm.logpdf(-1., x, 0.5), axis=-1)
    log_ws = pot_fn(xs)
    log_ws = log_ws - jax.scipy.special.logsumexp(log_ws)


    @jax.jit
    def r_multinomial():
        return multinomial(key, log_ws, xs)


    f_multinomial = lambda: r_multinomial()[1].block_until_ready()


    @jax.jit
    def r_soft():
        return soft_resampling(key, log_ws, xs, alpha=.5)


    f_soft = lambda: r_soft()[1].block_until_ready()


    @jax.jit
    def r_gumbel():
        return gumbel_softmax(key, log_ws, xs, tau=.5)


    f_gumbel = lambda: r_gumbel()[1].block_until_ready()

    print(f'Standard resampling with nsamples={nsamples}.')

    # Trigger JIT
    _ = f_multinomial()
    _ = f_soft()
    _ = f_gumbel()

    # Timeit
    for i in range(ntries):
        times_multinomial[i] = timeit.timeit(f_multinomial, number=1)
        times_soft[i] = timeit.timeit(f_soft, number=1)
        times_gumbel[i] = timeit.timeit(f_gumbel, number=1)

    # Save results
    np.save(f'./profiling_time/results/times-multinomial-{nsamples}', times_multinomial)
    np.save(f'./profiling_time/results/times-soft-{nsamples}', times_soft)
    np.save(f'./profiling_time/results/times-gumbel-{nsamples}', times_gumbel)

# Diffusion
for nsamples in nsampless:
    # Generate dummy data
    xs = jax.random.normal(key, shape=(nsamples, d))
    pot_fn = lambda x: jnp.sum(jax.scipy.stats.norm.logpdf(-1., x, 0.5), axis=-1)
    log_ws = pot_fn(xs)
    log_ws = log_ws - jax.scipy.special.logsumexp(log_ws)

    for nsteps in [4, 8, 16, 32]:
        print(f'Diffusion with nsamples={nsamples}, nsteps={nsteps}.')
        times_diffusion = np.zeros(ntries)
        ts = jnp.linspace(0., 1., nsteps + 1)


        @jax.jit
        def r_diffusion():
            return diffusion_resampling(key, log_ws, xs, -0.5, ts, integrator='euler', ode=True)


        f_diffusion = lambda: r_diffusion()[1].block_until_ready()

        # Trigger JIT
        _ = f_diffusion()

        # Timeit
        for i in range(ntries):
            times_diffusion[i] = timeit.timeit(f_diffusion, number=1)

        # Save results
        np.save(f'./profiling_time/results/times-diffusion-{nsamples}-{nsteps}', times_diffusion)

# OT
for nsamples in nsampless:
    # Generate dummy data
    xs = jax.random.normal(key, shape=(nsamples, d))
    pot_fn = lambda x: jnp.sum(jax.scipy.stats.norm.logpdf(-1., x, 0.5), axis=-1)
    log_ws = pot_fn(xs)
    log_ws = log_ws - jax.scipy.special.logsumexp(log_ws)

    for eps in [0.1, 0.2, 0.4, 0.8]:
        print(f'OT with nsamples={nsamples}, eps={eps}.')
        times_ot = np.zeros(ntries)


        @jax.jit
        def r_ot():
            return ensemble_ot(key, log_ws, xs, eps=eps)


        f_ot = lambda: r_ot()[1].block_until_ready()

        # Trigger JIT
        _ = f_ot()

        # Timeit
        for i in range(ntries):
            times_ot[i] = timeit.timeit(f_ot, number=1)

        # Save results
        np.save(f'./profiling_time/results/times-ot-{nsamples}-{eps}', times_ot)

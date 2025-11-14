import argparse
import timeit
import jax
import jax.numpy as jnp
import numpy as np
from diffres.resampling import multinomial, stratified, systematic, diffusion_resampling, soft_resampling, \
    gumbel_softmax, ensemble_ot
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--platform', type=str, default='cpu', help='The running platform.')
parser.add_argument('--integrator', type=str, default='jentzen_and_kloeden', help='The integrator.')
parser.add_argument('--sde', action='store_true', help='The probability flow model or the SDE model.')
args = parser.parse_args()

jax.config.update("jax_enable_x64", True)
key_main = jax.random.PRNGKey(666)

platform = args.platform
d = 8
y, xi_s = -0.5, 0.5
pot_fn = lambda x: jnp.sum(jax.scipy.stats.norm.logpdf(y, x, xi_s), axis=-1)
post_m = 1 / (1 + xi_s ** 2) * y

ntries = 50  # this is sufficient, as the running times (of different methods) are fairly different
nsampless = [128, 256, 512, 1024, 2048, 4096, 8192]

times_multinomial = np.zeros(ntries)
times_soft = np.zeros(ntries)
times_gumbel = np.zeros(ntries)
errs_multinomial = np.zeros((ntries, d))
errs_soft = np.zeros((ntries, d))
errs_gumbel = np.zeros((ntries, d))


def gen_data(key_):
    xs_ = jax.random.normal(key_, shape=(nsamples, d))
    log_ws_ = pot_fn(xs_)
    log_ws_ = log_ws_ - jax.scipy.special.logsumexp(log_ws_)
    return log_ws_, xs_


def compute_residual(log_ws, xs_):
    return post_m - jnp.einsum('n,nd->d', jnp.exp(log_ws), xs_)


for nsamples in nsampless:

    @jax.jit
    def r_multinomial(key_, log_ws_, xs_):
        return multinomial(key_, log_ws_, xs_)


    def f_multinomial(key_, log_ws_, xs_):
        vals = r_multinomial(key_, log_ws_, xs_)
        return vals[0].block_until_ready(), vals[1].block_until_ready()


    @jax.jit
    def r_soft(key_, log_ws_, xs_):
        return soft_resampling(key_, log_ws_, xs_, alpha=.5)


    def f_soft(key_, log_ws_, xs_):
        vals = r_soft(key_, log_ws_, xs_)
        return vals[0].block_until_ready(), vals[1].block_until_ready()


    @jax.jit
    def r_gumbel(key_, log_ws_, xs_):
        return gumbel_softmax(key_, log_ws_, xs_, tau=.5)


    def f_gumbel(key_, log_ws_, xs_):
        vals = r_gumbel(key_, log_ws_, xs_)
        return vals[0].block_until_ready(), vals[1].block_until_ready()


    print(f'Standard resampling with nsamples={nsamples}.')

    # Trigger JIT
    _ = f_multinomial(key_main, *gen_data(key_main))
    _ = f_soft(key_main, *gen_data(key_main))
    _ = f_gumbel(key_main, *gen_data(key_main))

    # Timeit and compute error
    key, _ = jax.random.split(key_main)
    for i in range(ntries):
        key, _ = jax.random.split(key)
        data = gen_data(key)

        times_multinomial[i] = timeit.timeit(partial(f_multinomial, key, *data), number=1)
        times_soft[i] = timeit.timeit(partial(f_soft, key, *data), number=1)
        times_gumbel[i] = timeit.timeit(partial(f_gumbel, key, *data), number=1)

        key, _ = jax.random.split(key)
        errs_multinomial[i] = compute_residual(*f_multinomial(key, *data))
        errs_soft[i] = compute_residual(*f_soft(key, *data))
        errs_gumbel[i] = compute_residual(*f_gumbel(key, *data))

    # Save results
    np.savez_compressed(f'./profiling_time/results/times-multinomial-{nsamples}-{platform}',
                        times=times_multinomial, errs=errs_multinomial)
    np.savez_compressed(f'./profiling_time/results/times-soft-{nsamples}-{platform}',
                        times=times_soft, errs=errs_soft)
    np.savez_compressed(f'./profiling_time/results/times-gumbel-{nsamples}-{platform}',
                        times=times_gumbel, errs=errs_gumbel)

# Diffusion
for nsamples in nsampless:
    dt = 0.1
    for nsteps in [4, 8, 16, 32]:
        print(f'Diffusion with nsamples={nsamples}, nsteps={nsteps}.')
        times_diffusion = np.zeros(ntries)
        errs_diffusion = np.zeros((ntries, d))
        ts = jnp.linspace(0., dt * nsteps, nsteps + 1)
        integrator = args.integrator
        ode = not args.sde


        @jax.jit
        def r_diffusion(key_, log_ws_, xs_):
            return diffusion_resampling(key_, log_ws_, xs_, -1., ts, integrator=integrator, ode=ode)


        def f_diffusion(key_, log_ws_, xs_):
            vals = r_diffusion(key_, log_ws_, xs_)
            return vals[0].block_until_ready(), vals[1].block_until_ready()


        # Trigger JIT
        _ = f_diffusion(key_main, *gen_data(key_main))

        # Timeit and compute error
        key, _ = jax.random.split(key_main)
        for i in range(ntries):
            key, _ = jax.random.split(key)
            data = gen_data(key)

            times_diffusion[i] = timeit.timeit(partial(f_diffusion, key, *data), number=1)

            key, _ = jax.random.split(key)
            errs_diffusion[i] = compute_residual(*f_diffusion(key, *data))

        # Save results
        np.savez_compressed(f'./profiling_time/results/times-diffusion-{nsamples}-{nsteps}-{platform}',
                            times=times_diffusion, errs=errs_diffusion)

# OT
for nsamples in nsampless:
    for eps in [0.1, 0.2, 0.4, 0.8]:
        print(f'OT with nsamples={nsamples}, eps={eps}.')
        times_ot = np.zeros(ntries)
        errs_ot = np.zeros((ntries, d))


        @jax.jit
        def r_ot(key_, log_ws_, xs_):
            return ensemble_ot(key_, log_ws_, xs_, eps=eps)


        def f_ot(key_, log_ws_, xs_):
            vals = r_ot(key_, log_ws_, xs_)
            return vals[0].block_until_ready(), vals[1].block_until_ready()


        # Trigger JIT
        _ = f_ot(key_main, *gen_data(key_main))

        # Timeit and compute error
        key, _ = jax.random.split(key_main)
        for i in range(ntries):
            key, _ = jax.random.split(key)
            data = gen_data(key)

            times_ot[i] = timeit.timeit(partial(f_ot, key, *data), number=1)

            key, _ = jax.random.split(key)
            errs_ot[i] = compute_residual(*f_ot(key, *data))

        # Save results
        np.savez_compressed(f'./profiling_time/results/times-ot-{nsamples}-{eps}-{platform}',
                            times=times_ot, errs=errs_ot)

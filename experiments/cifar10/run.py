import argparse
import jax
import jax.numpy as jnp
import optax
import numpy as np
from diffres.resampling import diffusion_resampling, multinomial_stopped, ensemble_ot, soft_resampling, gumbel_softmax
from diffres.feynman_kac import smc_feynman_kac, compute_ess
from diffres.data import CIFAR10
from diffres.nns import pbnn_cifar10
from diffres.tools import op_except_leading, accuracy

parser = argparse.ArgumentParser(description='Synthetic regression.')
parser.add_argument('--mc_id', type=int, default=0, help='The MC seed id.')

parser.add_argument('--nparticles', type=int, default=8, help='The number of SMC samples.')
parser.add_argument('--sg', type=float, default=.99, help='The transition semigroup.')
parser.add_argument('--depth', type=int, default=18, help='The resnet depth.')
parser.add_argument('--nsteps', type=int, default=2, help='The number of filtering steps.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--nepochs', type=int, default=200, help='The maximum number of iterations.')

parser.add_argument('--r', type=str, default='diffusion', help='The resampling method.')
parser.add_argument('--rthreshold', type=float, default=.5, help='The resampling threshold.')

parser.add_argument('--a', type=float, default=-0.5, help='The coefficient.')
parser.add_argument('--T', type=float, default=1., help='The diffusion terminal time.')
parser.add_argument('--dsteps', type=int, default=4, help='The integration steps of the diffusion.')
parser.add_argument('--integrator', type=str, default='jentzen_and_kloeden', help='The integrator.')
parser.add_argument('--sde', action='store_true', help='The probability flow model or the SDE model.')
parser.add_argument('--jitter', type=float, default=1e-5, help='The probability flow model or the SDE model.')

parser.add_argument('--tau', type=float, default=0.1, help='The gumbel temperature.')
parser.add_argument('--eps', type=float, default=1., help='The OT regulariser.')
parser.add_argument('--alpha', type=float, default=0.9, help='The softening parameter.')

args = parser.parse_args()

# Random key seed
# Separate the key branch for data and algorithm
mc_id = args.mc_id
key = np.load('rnd_keys.npy')[mc_id]
data_key, key = jax.random.split(key)

# Dataset creation
dataset = CIFAR10(data_key)
data_size = dataset.n
batch_size = args.batch_size
aug_fn = jax.jit(dataset.augmentation)

# Define the pBNN
key, _ = jax.random.split(key)
pbnn_phi, pbnn_psi, pbnn_forward_pass = pbnn_cifar10(key, batch_size, depth=args.depth)
shape_phi, shape_psi = pbnn_phi[0].shape, pbnn_psi[0].shape

# Define the FK model
sg = args.sg
nparticles = args.nparticles
nsteps = args.nsteps
resampling_threshold = args.rthreshold
chunk_size = batch_size * nsteps
shape_phis = (nparticles, *shape_phi)


def logpdf_likelihood(batch_data, phis_, psi_):
    xs_, ys_ = batch_data
    log_probs = jax.vmap(pbnn_forward_pass, in_axes=[0, None, None])(phis_, psi_, xs_)
    return op_except_leading(jnp.mean, jnp.take_along_axis(log_probs, ys_[:, 0][None, :, None], axis=2))


# Define the resampling
a = args.a
T = args.T
dsteps = args.dsteps
ts = jnp.linspace(0., T, dsteps + 1)
integrator = args.integrator
ode = not args.sde
jitter = args.jitter
tau = args.tau
alpha = args.alpha
eps = args.eps


def resampling(key_, log_ws_, samples_):
    if args.r == 'diffusion':
        return diffusion_resampling(key_, log_ws_, samples_, a, ts, integrator=integrator, ode=ode, jitter=jitter)
    elif args.r == 'ot':
        return ensemble_ot(key_, log_ws_, samples_, eps, implicit_diff=False)
    elif args.r == 'gumbel':
        return gumbel_softmax(key_, log_ws_, samples_, tau)
    elif args.r == 'soft':
        return soft_resampling(key_, log_ws_, samples_, alpha)
    else:
        return multinomial_stopped(key_, log_ws_, samples_)


def resampling_routine(key_, log_ws_, samples_):
    ess = compute_ess(log_ws_)
    return jax.lax.cond(ess < resampling_threshold * nparticles,
                        lambda _: (resampling(key_, log_ws_, samples_)),
                        lambda _: (log_ws_, samples_),
                        None)


# Define the filter
def pf(key_, psi_, data_chunk_, posterior_):
    key_r, key_pf = jax.random.split(key_)
    posterior_ = resampling_routine(key_r, *posterior_)

    def m0(key__, _):
        return sg * posterior_[1] + (1 - sg ** 2) ** 0.5 * jax.random.normal(key__, shape=shape_phis)

    def log_g0(phis0, batch_data):
        return posterior_[0] + logpdf_likelihood(batch_data, phis0, psi_)

    def m_log_g(key__, phis_km1, batch_data):
        phis_k = sg * phis_km1 + (1 - sg ** 2) ** 0.5 * jax.random.normal(key__, shape=shape_phis)
        return logpdf_likelihood(batch_data, phis_k, psi_), phis_k

    data_chunk_ = jax.tree_util.tree_map(lambda x: x.reshape((nsteps, batch_size, -1)),
                                         data_chunk_)
    return smc_feynman_kac(key_, m0, log_g0, m_log_g, data_chunk_, nparticles, nsteps - 1,
                           resampling=resampling, resampling_threshold=resampling_threshold,
                           return_path=False)


# Define the loss and opt
def loss_fn(psi_, key_, data_chunk_, posterior_):
    samples_, log_ws_, nll_, ess_ = pf(key_, psi_, data_chunk_, posterior_)
    scaled_nll = nll_ / nsteps
    return scaled_nll, ((log_ws_, samples_), ess_)


@jax.jit
def step(psi_, opt_state_, key_, data_chunk_, posterior_):
    (loss_, aux_out_), grad = jax.value_and_grad(loss_fn, has_aux=True)(psi_, key_, data_chunk_, posterior_)
    updates, opt_state_ = optimiser.update(grad, opt_state_, psi_)
    psi_ = optax.apply_updates(psi_, updates)
    return psi_, opt_state_, loss_, aux_out_


# Caching rule
@jax.jit
def val_metric(log_ws_, samples_, psi_):
    def acc_(s):
        preds = pbnn_forward_pass(s, psi_, dataset.val_xs)
        return accuracy(preds, jax.nn.one_hot(dataset.val_ys[:, 0], 10))

    return jnp.dot(jnp.exp(log_ws_), jax.vmap(acc_)(samples_))


# Optimisation setup
total_niters = args.nepochs * (data_size // chunk_size)
warmup_niters = 10 * (data_size // chunk_size)
lr_schedule = optax.warmup_cosine_decay_schedule(init_value=0., peak_value=1e-2, warmup_steps=warmup_niters,
                                                 decay_steps=total_niters, end_value=1e-5)
optimiser = optax.chain(optax.clip_by_global_norm(1.0),
                        optax.adam(learning_rate=lr_schedule))
psi = pbnn_psi[0]
opt_state = optimiser.init(psi)

# Optimisation loop
val_best = 0.
val_current = 0.
losses_train = np.zeros(total_niters)
losses_val = np.zeros(total_niters)
esss_train = np.zeros(total_niters)

posterior = (-jnp.log(nparticles) * jnp.ones(nparticles),
             jnp.tile(pbnn_phi[0][None, ...], (nparticles, 1)))
for i in range(args.nepochs):
    data_key, key_aug = jax.random.split(data_key)
    dataset.init_enumeration(data_key, chunk_size)
    for j in range(data_size // chunk_size):
        data_chunk = dataset.enumerate_subset(j)
        data_chunk = aug_fn(key_aug, data_chunk)
        key, _ = jax.random.split(key)
        psi, opt_state, loss, aux_out = step(psi, opt_state, key, data_chunk, posterior)
        posterior = aux_out[0]
        mean_ess = np.mean(aux_out[-1])

        # Check fo every 5 iters
        if (j + 1) % 5 == 0:
            val_current = val_metric(*posterior, psi)
            if val_current > val_best:
                val_best = val_current
                np.savez(f'./cifar10/checkpoints/checkpoint-cifar10-best',
                         log_ws=posterior[0], samples=posterior[1], psi=psi, i=i, j=j)

        idx = i * (data_size // chunk_size) + j
        losses_train[idx] = loss
        losses_val[idx] = val_current
        esss_train[idx] = mean_ess

        print(f'Epoch {i} / {args.nepochs} '
              f'| Iter {j} / {data_size // chunk_size} '
              f'| Train loss {loss:.5f} | ess {mean_ess:.3f} '
              f'| Val current {val_current:.3f} '
              f'| Val best {val_best:.3f}')

import argparse
import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
import sys
import matplotlib.pyplot as plt
from flax import nnx
from pathlib import Path
from diffres.resampling import (multinomial, stratified, systematic, diffusion_resampling,
                                 multinomial_stopped, ensemble_ot, soft_resampling, gumbel_softmax)
from diffres.feynman_kac import smc_feynman_kac
from diffres.nns import nnx_save, nnx_load, NNWeather_dynamics
#bash demos/temperature/run_pendulum_decoder.sh

parser = argparse.ArgumentParser()
parser.add_argument('--mc_id',            type=int,   default=1)
parser.add_argument('--nparticles',       type=int,   default=32)
parser.add_argument('--lr',               type=float, default=5e-3)
parser.add_argument('--niters',           type=int,   default=1000)
parser.add_argument('--a',                type=float, default=-0.5)
parser.add_argument('--T',                type=float, default=1.)
parser.add_argument('--dsteps',           type=int,   default=8)
parser.add_argument('--integrator',       type=str,   default='euler')
parser.add_argument('--sde',              action='store_true')
parser.add_argument('--eps_rs',           type=float, default=None)
parser.add_argument('--alpha_rs',         type=float, default=1.)
parser.add_argument('--tau_rs',           type=float, default=0.1)
parser.add_argument('--resampler_id',     type=str,   default='0')
parser.add_argument('--sigma_q_scale',    type=float, default=0.1)
parser.add_argument('--training_noise',   type=float, default=0.01)
parser.add_argument('--nll_scale_factor', type=float, default=1.)
parser.add_argument('--experiment_id',    type=str,   default='test_experiment')
parser.add_argument('--dataset',          type=str,   default='cloud',
                    help='Which dataset to use: cloud or temperature')
parser.add_argument('--obs_mode',         type=str,   default='pretrain',
                    help='pretrain (99% observed) or finetune (50% observed)')
parser.add_argument('--nsteps', type=int, default=256,
                    help='Number of steps to run the particle filter')

args = parser.parse_args()

mc_id = args.mc_id
jax.config.update("jax_enable_x64", False)
key = np.load('rnd_keys.npy')[mc_id]

demo_id        = args.experiment_id
results_dir    = os.path.join('demos', 'temperature', 'results',     demo_id)
checkpoint_dir = os.path.join('demos', 'temperature', 'checkpoints', demo_id)
dynamics_dir   = os.path.join(results_dir, 'dynamics_eval')
os.makedirs(dynamics_dir,   exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

obs_dir = Path('demos') / 'temperature' / 'observations2' / args.dataset

os.makedirs(results_dir, exist_ok=True)
log_path = os.path.join(results_dir, 'training_log.txt')

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log      = open(filepath, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_path)

# load data
sequence     = np.load(obs_dir / 'sequence.npy')                         # (T, H, W, C) = (257, 32, 64, 1)
mask         = np.load(obs_dir / f'mask_{args.obs_mode}.npy')            # (H, W, C) = (32, 64, 1) boolean
measurements = np.load(obs_dir / f'measurements_{args.obs_mode}.npy')    # (T, N) = (257, 2031), N=number of observed pixels

nsteps       = min(args.nsteps, sequence.shape[0] - 1)
sequence     = sequence[:nsteps + 1]
measurements = measurements[:nsteps + 1]

nsteps, H, W, C = sequence.shape[0] - 1, *sequence.shape[1:]
N = mask.sum()  # number of observed pixels

print(f'Loaded sequence: {sequence.shape}')
print(f'Mask ({args.obs_mode}): {mask.shape}, N={N} observed pixels')
print(f'Measurements: {measurements.shape}')

print(f'mask dtype: {mask.dtype}')
print(f'sequence range: [{sequence.min():.3f}, {sequence.max():.3f}]')
print(f'measurements range: [{measurements.min():.3f}, {measurements.max():.3f}]')

# set model parameters
sigma_q_scale    = args.sigma_q_scale
sigma_q_std      = jnp.sqrt(sigma_q_scale)   # scalar std for noise image
training_noise   = args.training_noise
nll_scale_factor = args.nll_scale_factor
nparticles       = args.nparticles

# set initial state to first image in sequence
x0 = sequence[0]   # (H, W, C)

# set resampling parameters
a           = args.a
T           = args.T
dsteps      = args.dsteps
ts          = jnp.linspace(0., T, dsteps + 1)
integrator  = args.integrator
ode         = not args.sde
eps_rs      = args.eps_rs
alpha_rs    = args.alpha_rs
tau_rs      = args.tau_rs

resampling_threshold = 2.0

print_prefix    = f'Weather ({mc_id}) | a {a} | T {T} | dsteps={dsteps} | {integrator} {"| ode" if ode else "| sde"}'
filename_prefix = f'weather-{a}-{T}-{dsteps}-{integrator}-{"ode" if ode else "sde"}-'

def resampling_diffusion(key_, log_ws_, samples_):
    return diffusion_resampling(key_, log_ws_, samples_, a, ts, integrator=integrator, ode=ode, jitter=1e-5)

def resampling_ot(key_, log_ws_, samples_):
    return ensemble_ot(key_, log_ws_, samples_, eps_rs, implicit_diff=False)

def resampling_gumbel(key_, log_ws_, samples_):
    return gumbel_softmax(key_, log_ws_, samples_, tau=tau_rs)

def resampling_soft(key_, log_ws_, samples_):
    return soft_resampling(key_, log_ws_, samples_, alpha=alpha_rs)

select_resampler = {
    '0': resampling_diffusion,
    '1': resampling_ot,
    '2': resampling_gumbel,
    '3': resampling_soft,
}
resampling = select_resampler[args.resampler_id]

# initialize all particles to x0
def m0_sampler(key_, _):
    return jnp.ones((nparticles, H, W, C)) * x0

# initialize NN dynamics model
key, subkey = jax.random.split(key)
model = NNWeather_dynamics(
    img_channels=C,
    channel_widths=[32, 64, 128],  # [8, 16, 32]
    rngs=nnx.Rngs(subkey)
)

# load checkpoint for warm start
#PRETRAIN_CHECKPOINT = 'demos/temperature/checkpoints/pretrain_unet11/pretrained_mc1_iter_0999'
#model = nnx_load(model, PRETRAIN_CHECKPOINT)
#print(f'Loaded pretrained checkpoint: {PRETRAIN_CHECKPOINT}')

# set optimiser setting
inner_opt      = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
optax_optimizer = optax.apply_if_finite(inner_opt, max_consecutive_errors=10)
optimiser      = nnx.Optimizer(model, optax_optimizer, wrt=nnx.Param)

# functions to visualize the filter progress
cmap_obj = plt.get_cmap('Blues' if args.dataset == 'cloud' else 'RdBu_r').copy()
cmap_obj.set_bad(color='black')

# reconstruct (H, W, C) image from flattened measurements vector, set masked pixels to nan
def measurement_to_image(y_k, mask_):
    img = np.full(mask_.shape, np.nan, dtype=np.float32)
    img[np.array(mask_)] = np.array(y_k)
    return img

def save_reconstruction_grid(sequence_, measurements_, mask_, particle_mean_seq,
                              nsteps_, save_dir, filename, n_frames=9):
    """
    3-row grid for n_frames evenly spaced timesteps:
      First row: ground truth image
      2nd row:   measurement image (observed pixels only, masked pixels in black)
      3rd row:   particle mean (reconstruction)
    """
    os.makedirs(save_dir, exist_ok=True)
    indices = np.linspace(0, nsteps_, n_frames, dtype=int)
    fig, axes = plt.subplots(3, n_frames, figsize=(n_frames * 2.5, 7))

    row_titles = ['Ground truth',
                  'Measurement',
                  'Particle mean']

    for row, title in enumerate(row_titles):
        axes[row, 0].set_ylabel(title, fontsize=9)

    for col, idx in enumerate(indices):
        # ground truth
        axes[0, col].imshow(np.array(sequence_[idx, :, :, 0]),
                            cmap=cmap_obj, origin='lower', vmin=0, vmax=1)
        axes[0, col].set_title(f't={idx}', fontsize=7)
        axes[0, col].axis('off')

        # measurement image (masked pixels in black)
        meas_img = measurement_to_image(measurements_[idx], mask_)
        axes[1, col].imshow(meas_img[:, :, 0],
                            cmap=cmap_obj, origin='lower', vmin=0, vmax=1)
        axes[1, col].axis('off')

        # mean reconstruction
        axes[2, col].imshow(np.array(particle_mean_seq[idx, :, :, 0]),
                            cmap=cmap_obj, origin='lower', vmin=0, vmax=1)
        axes[2, col].axis('off')

    plt.suptitle('Reconstruction', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)

# plot absolute error |ground_truth - mean_reconstruction| on masked pixels only
# black: correct, red: moderate error, yellow: large error, white: max error
def save_error_grid(sequence_, particle_mean_seq, mask_,
                    nsteps_, save_dir, filename, n_frames=9):

    os.makedirs(save_dir, exist_ok=True)
    indices = np.linspace(0, nsteps_, n_frames, dtype=int)
    fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 2.5, 2.5))

    for col, idx in enumerate(indices):
        err = np.abs(np.array(sequence_[idx]) - np.array(particle_mean_seq[idx]))
        err_masked = np.full(mask_.shape, np.nan, dtype=np.float32)
        err_masked[~np.array(mask_)] = err[~np.array(mask_)]  # only unobserved pixels
        axes[col].imshow(err_masked[:, :, 0], cmap='hot', origin='lower', vmin=0, vmax=0.5)
        axes[col].set_title(f't={idx}', fontsize=7)
        axes[col].axis('off')

    plt.suptitle('Abs error on masked pixels', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)

# define loss function
def loss_fn(model_: NNWeather_dynamics, key_):

    def logpdf_y_cond_x_NN_(y, x):
        """
        y: flattened measurement vector (N,) 
        x: current particle states (nparticles, H, W, C)
        returns: log likelihoods (nparticles,)
        """
        pred = jax.vmap(lambda xi: xi[mask].flatten())(x) # (nparticles, N), y shape (N,)
        log_probs = jax.scipy.stats.norm.logpdf(y, loc=pred, scale=training_noise) # (nparticles, N)
        
        #jax.debug.print("pred min={} max={} finite={}", 
        #            jnp.min(pred), jnp.max(pred), jnp.all(jnp.isfinite(pred)))
        #jax.debug.print("y min={} max={} finite={}", 
        #            jnp.min(y), jnp.max(y), jnp.all(jnp.isfinite(y)))
        #jax.debug.print("log_probs min={} max={}", jnp.min(log_probs), jnp.max(log_probs))
        
        return jnp.sum(log_probs, axis=1) / N # (nparticles,) normalized by number of observed pixels N

    def log_g0_NN_(samples, y0_):
        return logpdf_y_cond_x_NN_(y0_, samples)

    def m_log_g_NN_(key__, samples, y):
        # shape qs_, samples, prop_samples: (nparticles, H, W, C)
        qs_ = jax.random.normal(key__, shape=(nparticles, H, W, C)) * sigma_q_std # one noise image per particle
        prop_samples = jax.vmap(model_.f_dynamics)(samples, qs_)
        return logpdf_y_cond_x_NN_(y, prop_samples), prop_samples

    _, log_ws_, nll, ess, norm_sum = smc_feynman_kac(
        key_,
        m0_sampler,
        log_g0_NN_,
        m_log_g_NN_,
        measurements, # (T, N) flattened observation vector, not full images
        nparticles,
        nsteps,
        resampling=resampling,
        resampling_threshold=resampling_threshold,
        return_path=False
    )

    nll_loss   = nll / N #(nsteps * N)
    norm_loss  = 0.0 * norm_sum
    total_loss = nll_loss + norm_loss
    return total_loss, (nll_loss, ess, norm_loss)

# 
@nnx.jit
def train_step(model_, optimiser_, key_):
    (loss_, (nll_loss_, ess_, norm_loss_)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model_, key_)
    optimiser_.update(model_, grads)
    return loss_, ess_, nll_loss_, norm_loss_

# simulate rollout of dynamics from x0
def simulate_dynamics_only(key_, dynamics_func, x_init, num_steps, use_noise=True):
    def scan_body(carry_x, q_k):
        next_x = dynamics_func(carry_x, q_k)
        return next_x, next_x
    if use_noise:
        key_q, _ = jax.random.split(key_)
        qs = jax.random.normal(key_q, shape=(num_steps, H, W, C)) * sigma_q_std
    else:
        qs = jnp.zeros((num_steps, H, W, C))
    _, xs_ = jax.lax.scan(scan_body, x_init, qs)
    return jnp.concatenate([x_init[jnp.newaxis, ...], xs_], axis=0) # (num_steps+1, H, W, C)

# PF training loop
losses = np.zeros(args.niters)

for i in range(args.niters):
    key, subkey = jax.random.split(key)
    loss, ess, nll, norm_loss = train_step(model, optimiser, subkey)
    losses[i] = loss

    res_frac = jnp.sum(ess < resampling_threshold * nparticles)
    print(f' | Iter {i} | loss {loss:.4f} | nll {nll:.4f} | mean ess {jnp.mean(ess):.1f} | num_res {res_frac}')

    if i % 10 == 0:
        num_rollouts = 16  # number of noisy rollouts to average
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, num_rollouts)
        xs_pred = jnp.mean(jax.vmap(lambda k: simulate_dynamics_only(k, model.f_dynamics, x0, nsteps, use_noise=True))(keys), axis=0) # (T, H, W, C)

        #xs_pred = simulate_dynamics_only(subkey, model.f_dynamics, x0, nsteps, use_noise=False) # rolled out learned dynamics from x0 (no noise for clean comparison)
        
        # plot reconstruction grid: ground truth vs observations vs prediction (averaged rollouts)
        filename = filename_prefix + f'{mc_id}' + f'_reconstruction_iter_{i:04d}.png'
        save_reconstruction_grid(
            sequence_=sequence,
            measurements_=measurements,
            mask_=mask,
            particle_mean_seq=xs_pred,
            nsteps_=nsteps,
            save_dir=dynamics_dir,
            filename=filename,
        )

        # plot to visualize preciction error (absolute error) at unobserved pixels
        err_filename = filename_prefix + f'{mc_id}' + f'_error_iter_{i:04d}.png'
        save_error_grid(
            sequence_=sequence,
            particle_mean_seq=xs_pred,
            mask_=mask,
            nsteps_=nsteps,
            save_dir=dynamics_dir,
            filename=err_filename,
        )

        # compute pixel MSE on unobserved pixels only
        mask_broadcast = jnp.broadcast_to(mask, sequence.shape)
        masked_mse = jnp.mean((xs_pred[~mask_broadcast] - sequence[~mask_broadcast]) ** 2)
        print(f'Masked pixel MSE: {masked_mse:.4f}')

# save checkpoint
checkpoint_path = os.path.join(checkpoint_dir, filename_prefix + f'{mc_id}')
nnx_save(model, checkpoint_path)

# plor loss curve
plt.figure()
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training loss')
loss_save_path = os.path.join(results_dir, 'weather_loss_vs_iter.png')
plt.savefig(loss_save_path)
plt.close()
print(f'Saved loss curve to {loss_save_path}')

# final evaluation
key, _ = jax.random.split(key)
num_rollouts = 100  # number of noisy rollouts to average
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num_rollouts)
xs_pred_final = jnp.mean(jax.vmap(lambda k: simulate_dynamics_only(k, model.f_dynamics, x0, nsteps, use_noise=True))(keys), axis=0) # (T, H, W, C)

#xs_pred_final = simulate_dynamics_only(key, model.f_dynamics, x0, nsteps, use_noise=False)

# RMSE on masked pixels across all timesteps
mask_broadcast = jnp.broadcast_to(mask, sequence.shape)
masked_rmse = jnp.sqrt(jnp.mean((xs_pred_final[~mask_broadcast] - sequence[~mask_broadcast]) ** 2))
print(print_prefix + f' | Final masked-pixel RMSE: {masked_rmse:.4f}')

# final reconstruction grid
save_reconstruction_grid(
    sequence_=sequence,
    measurements_=measurements,
    mask_=mask,
    particle_mean_seq=xs_pred_final,
    nsteps_=nsteps,
    save_dir=results_dir,
    filename='final_reconstruction.png',
)

# final error grid
save_error_grid(
    sequence_=sequence,
    particle_mean_seq=xs_pred_final,
    mask_=mask,
    nsteps_=nsteps,
    save_dir=results_dir,
    filename='final_error.png',
)

# save results
save_path = os.path.join(results_dir, 'saved_data.npz')
np.savez_compressed(
    save_path,
    losses=losses,
    sequence=np.array(sequence),
    measurements=np.array(measurements),
    mask=np.array(mask),
    masked_rmse=float(masked_rmse),
    mc_id=mc_id,
    nsteps=nsteps,
    nparticles=nparticles,
    lr=args.lr,
    niters=args.niters,
    a=a,
    T=T,
    dsteps=dsteps,
    integrator=integrator,
    sde=args.sde,
    eps_rs=eps_rs if eps_rs is not None else -1,
    alpha_rs=alpha_rs,
    tau_rs=tau_rs,
    resampler_id=args.resampler_id,
    sigma_q_scale=sigma_q_scale,
    training_noise=training_noise,
    nll_scale_factor=nll_scale_factor,
    dataset=args.dataset,
    obs_mode=args.obs_mode,
)
print(f'Saved results to {save_path}')
import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
import sys
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flax import nnx
from pathlib import Path
from diffres.nns import nnx_save, nnx_load, NNWeather_dynamics

DATASET           = 'cloud'
NITERS            = 1000
LR                = 1e-3
MC_ID             = 1
EXPERIMENT_ID     = 'pretrain_unet11'
CHECKPOINT_EVERY  = 20
PLOT_EVERY        = 20
LOAD_CHECKPOINT   = None
SIGMA_Q_STD       = jnp.sqrt(0.1)  # ! note: must match sigma_q_std in particle filter script

script_dir     = Path(__file__).parent
obs_dir        = script_dir / 'observations' / DATASET
results_dir    = script_dir / 'results' / EXPERIMENT_ID
checkpoint_dir = script_dir / 'checkpoints' / EXPERIMENT_ID
os.makedirs(results_dir,    exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

log_path = results_dir / 'pretrain_log.txt'

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log      = open(filepath, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_path)
print(f'Logging to {log_path}')

sequence = jnp.array(np.load(obs_dir / 'sequence.npy'),     dtype=jnp.float32)  # (T, H, W, C)
mask     = jnp.array(np.load(obs_dir / 'mask_pretrain.npy'))                    # (H, W, C)

T, H, W, C = sequence.shape
nsteps     = T-1

print(f'Sequence: {sequence.shape}')
print(f'Mask (pretrain): {mask.shape}, N={mask.sum()} observed')

jax.config.update("jax_enable_x64", False)
key = np.load('rnd_keys.npy')[MC_ID]
key = jax.random.PRNGKey(int(key[0]))

key, subkey = jax.random.split(key)
model = NNWeather_dynamics(
    img_channels=C,
    channel_widths=[32, 64, 128],
    rngs=nnx.Rngs(subkey)
)

if LOAD_CHECKPOINT is not None:
    model = nnx_load(model, LOAD_CHECKPOINT)
    print(f'Loaded checkpoint: {LOAD_CHECKPOINT}')
else:
    print('Training from scratch')

inner_opt       = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR))
optax_optimizer = optax.apply_if_finite(inner_opt, max_consecutive_errors=10)
optimiser       = nnx.Optimizer(model, optax_optimizer, wrt=nnx.Param)

@nnx.jit
def pretrain_step(model_, opt_, x_prev, x_next, key_):
    q = jax.random.normal(key_, shape=x_prev.shape) * SIGMA_Q_STD
    def mse_loss(m):
        pred = m.f_dynamics(x_prev, q)
        return jnp.mean((pred - x_next) ** 2)
    loss, grads = nnx.value_and_grad(mse_loss)(model_)
    opt_.update(model_, grads)
    return loss

cmap_obj = plt.get_cmap('Blues' if DATASET == 'cloud' else 'RdBu_r').copy()
cmap_obj.set_bad(color='black')

def save_comparison_grid(sequence_, model_, indices, savepath, n_frames=8):
    fig, axes = plt.subplots(2, n_frames, figsize=(n_frames * 2.5, 5))

    def scan_body(carry, _):
        x      = carry
        q      = jnp.zeros_like(x)
        x_next = model_.f_dynamics(x, q)
        return x_next, x_next

    _, xs_pred = jax.lax.scan(scan_body, sequence_[0], None, length=nsteps)
    xs_pred    = jnp.concatenate([sequence_[0:1], xs_pred], axis=0)  # (T, H, W, C)

    row_labels = ['Ground truth', 'UNet prediction']
    for row, (imgs, label) in enumerate(zip([sequence_, xs_pred], row_labels)):
        axes[row, 0].set_ylabel(label, fontsize=9)
        for col, idx in enumerate(indices):
            img = np.array(imgs[idx, :, :, 0])
            axes[row, col].imshow(img, cmap=cmap_obj, origin='lower', vmin=0, vmax=1)
            axes[row, col].set_title(f't={idx}', fontsize=7)
            axes[row, col].axis('off')

    plt.suptitle('Pre-training reconstruction', fontsize=11)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved')

indices = np.linspace(0, nsteps, 8, dtype=int)

key, subkey = jax.random.split(key)
_ = pretrain_step(model, optimiser, sequence[0], sequence[1], subkey)

losses = []

print('Starting training\n')
print(f'Checkpoints every {CHECKPOINT_EVERY} iters')
print(f'Plots every {PLOT_EVERY} iters')
print(f'Noise sigma_q_std={SIGMA_Q_STD}\n')

for i in range(NITERS):
    key, subkey = jax.random.split(key)
    perm        = np.array(jax.random.permutation(subkey, nsteps))

    iter_losses = []
    for t in perm:
        key, subkey = jax.random.split(key)
        x_prev = sequence[t]
        x_next = sequence[t + 1]
        loss   = pretrain_step(model, optimiser, x_prev, x_next, subkey)
        iter_losses.append(float(loss))

    mean_loss = np.mean(iter_losses)
    losses.append(mean_loss)
    print(f'Iter {i:4d} | MSE loss {mean_loss:.6f}')

    if i % CHECKPOINT_EVERY == 0 or i == NITERS - 1:
        checkpoint_path = str(checkpoint_dir / f'pretrained_mc{MC_ID}_iter_{i:04d}')
        nnx_save(model, checkpoint_path)
        print(f'Checkpoint saved')

    if i % PLOT_EVERY == 0 or i == NITERS - 1:
        save_comparison_grid(
            sequence_=sequence,
            model_=model,
            indices=indices,
            savepath=results_dir / f'pretrain_comparison_iter_{i:04d}.png',)
            
plt.figure()
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.title('Pre-training loss')
plt.yscale('log')
plt.grid(True)
plt.savefig(results_dir / 'pretrain_loss.png', dpi=150)
plt.close()
print(f'\nSaved loss curve')
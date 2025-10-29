import argparse
import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
import matplotlib.pyplot as plt
from flax import nnx
from diffres.resampling import (multinomial, stratified, systematic, diffusion_resampling, multinomial_stopped,
                                ensemble_ot, soft_resampling, gumbel_softmax)
from diffres.feynman_kac import smc_feynman_kac
from diffres.nns import NNPendulum_decoder_SSM, nnx_save
from diffres.tools import leading_concat

parser = argparse.ArgumentParser()
parser.add_argument('--mc_id', type=int, default=1, help='The MC id.')
parser.add_argument('--nsteps', type=int, default=256, help='Number of time steps.')
parser.add_argument('--mT', type=float, default=4., help='The model end time.')
parser.add_argument('--nparticles', type=int, default=32, help='Number of nparticles.')
parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate.')
parser.add_argument('--niters', type=int, default=1000, help='Number of learning iterations.')
parser.add_argument('--npreds', type=int, default=100, help='Number of ensemble predictions.')
parser.add_argument('--a', type=float, default=-0.5, help='The coefficient.')
parser.add_argument('--T', type=float, default=1., help='The diffusion terminal time.')
parser.add_argument('--dsteps', type=int, default=8, help='The integration steps of the diffusion.')
parser.add_argument('--integrator', type=str, default='euler', help='The integrator.')
parser.add_argument('--sde', action='store_true', help='The probability flow model or the SDE model.')
args = parser.parse_args()

mc_id = args.mc_id
jax.config.update("jax_enable_x64", True)  # TODO:
key = np.load('rnd_keys.npy')[mc_id]

# Model parameters
dx = 2
g_true = 9.81
pendulum_length = 0.4
sigma_q = jnp.array([0.01, 0.01])  # TODO: Double check if this is scale/var
sigma_xi_pixel = 0.01
img_height, img_width, img_channels = 32, 32, 1

t0 = 0.
nsteps = args.nsteps
mT = args.mT
dt = mT / nsteps


def f(x, q):
    alpha, ddt_alpha = x
    return jnp.array(
        [alpha + ddt_alpha * dt + q[0], ddt_alpha - (g_true / pendulum_length) * jnp.sin(alpha) * dt + q[1]])


def g_true_renderer(x, img_height, img_width, dpi=50):
    """
    Renders an image of a pendulum from its state.

    Input: x: state of shape (2,) containing angle and angular velocity
    Output: numpy array of shape (img_height, img_width, 1) representing the greyscale image with pixel values normalized to (0,1).
    """
    # calculate position of pendulum
    alpha = x[0]
    x_centre, y_centre = 0.5, 0.5
    x_pos = x_centre + pendulum_length * jnp.sin(alpha)
    y_pos = y_centre - pendulum_length * jnp.cos(alpha)

    # create plot
    fig_width, fig_height = img_width / dpi, img_height / dpi
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor('black')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('black')
    ax.plot([x_centre, x_pos], [y_centre, y_pos], color='white', lw=2)
    ax.plot(x_pos, y_pos, 'o', color='white', markersize=6)

    # render image
    fig.canvas.draw()
    img_data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    plt.close(fig)

    # convert to grayscale (0-255) and normalize to range (0,1)
    img_array_argb = img_data.reshape(img_height, img_width, 4)
    img_array_rgb = img_array_argb[..., 1:]
    img_array_gray = np.mean(img_array_rgb / 255.0, axis=2, keepdims=True)

    return img_array_gray.astype(np.float32)


def save_image_grid(ys_images: np.ndarray,
                    nsteps: int,
                    save_dir: str = "observations",
                    filename: str = "training_data.png"):
    """
    Saves a 3x3 grid of sample images from the observation sequence.
    """
    os.makedirs(save_dir, exist_ok=True)
    indices_to_plot = jnp.linspace(0, nsteps, 9, dtype=jnp.int32)  # select 9 frames to plot
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        idx = indices_to_plot[i]
        img = np.asarray(ys_images[idx])
        ax.imshow(img.squeeze(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Frame {idx}")
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 1])
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close(fig)


def save_comparison_grid(ys_true, ys_pred, nsteps,
                         save_dir="evaluation",
                         filename="decoder_comparison.png"):
    """
    Saves a 3x6 grid comparing true vs. predicted images.
    """
    os.makedirs(save_dir, exist_ok=True)
    indices_to_plot = jnp.linspace(0, nsteps, 9, dtype=jnp.int32)
    fig, axes = plt.subplots(3, 6, figsize=(12, 6))

    for i in range(9):
        row, col_idx = divmod(i, 3)
        idx = indices_to_plot[i]

        img_true = np.asarray(ys_true[idx]).squeeze()
        ax_true = axes[row, col_idx * 2]
        ax_true.imshow(img_true, cmap='gray', vmin=0, vmax=1)
        ax_true.set_title(f"True (Frame {idx})")
        ax_true.axis('off')

        img_pred = np.asarray(ys_pred[idx]).squeeze()
        ax_pred = axes[row, col_idx * 2 + 1]
        ax_pred.imshow(img_pred, cmap='gray', vmin=0, vmax=1)
        ax_pred.set_title(f"Pred (Frame {idx})")
        ax_pred.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 1])
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close(fig)


# Simulate model
x0 = jnp.array([jnp.pi / 2, 0.0])


def simulate_hidden_states(key_, f_, x0_, nsteps_):
    def scan_body(carry, elem):
        x = carry
        q_k = elem
        x = f_(x, q_k)
        return x, x

    key_q, _ = jax.random.split(key_)
    qs = jax.random.normal(key_q, shape=(nsteps_, dx)) * sigma_q
    _, xs_ = jax.lax.scan(scan_body, x0_, qs)
    return leading_concat(x0_, xs_)


def render_observations(xs, renderer_f, noise_scale=None):
    nsteps_ = xs.shape[0] - 1
    ys_images = []

    rng = np.random.default_rng(seed=42)  # Adapt this to jax
    noise = 0.0
    for k in range(nsteps_ + 1):
        clean_image = renderer_f(xs[k], img_height, img_width)
        if noise_scale is not None:
            noise = rng.normal(loc=0.0, scale=noise_scale, size=clean_image.shape)
        noisy_image = np.clip(clean_image + noise, 0.0, 1.0)
        ys_images.append(noisy_image)

    return np.stack(ys_images, axis=0)


key, _ = jax.random.split(key)
xs = simulate_hidden_states(key, f, x0, nsteps)
xs_np = np.array(xs)

ys_imgs = render_observations(xs_np, g_true_renderer,
                              noise_scale=sigma_xi_pixel)  # shape (nsteps+1, img_height, img_width, 1)
ys = jnp.array(ys_imgs)
print(f"Generated data with shape xs: {xs.shape}, ys: {ys.shape}")

# set up PF
nparticles = args.nparticles
a = args.a
T = args.T
dsteps = args.dsteps
ts = jnp.linspace(0., T, dsteps + 1)
integrator = args.integrator
ode = not args.sde
resampling_threshold = 1.

print_prefix = f'Diffres ({mc_id}) | a {a} | T {T} | dsteps={dsteps} | {integrator} {"| ode" if ode else "| sde"}'
filename_prefix = f'diffres-{a}-{T}-{dsteps}-{integrator}-{"ode" if ode else "sde"}-'

save_dir = 'experiments/pendulum/observations/'
filename = filename_prefix + f'{mc_id}' + '_observation_seq.png'
save_image_grid(ys_imgs, nsteps, save_dir=save_dir, filename=filename)


def resampling(key_, log_ws_, samples_):
    return diffusion_resampling(key_, log_ws_, samples_, a, ts, integrator=integrator, ode=ode)


# def resampling(key_, log_ws_, samples_):
#     return ensemble_ot(key_, log_ws_, samples_, 0.8)


# def resampling(key_, log_ws_, samples_):
#     return gumbel_softmax(key_, log_ws_, samples_, tau=0.1)


def m0_sampler(key_, _):
    return jnp.ones((nparticles, dx)) * x0


# Learn NN representation of full SSM (hidden dynamics model f and observation model g)
key, _ = jax.random.split(key)
model = NNPendulum_decoder_SSM(dt=dt, rngs=nnx.Rngs(key))
optimiser = optax.lion(args.lr)
optimiser = nnx.Optimizer(model, optimiser, wrt=nnx.Param)
# optax_optimizer = optax.chain(
#    optax.clip_by_global_norm(1.0),  
#    optax.lion(args.lr)            
# )
# optimiser = nnx.Optimizer(model, optax_optimizer, wrt=nnx.Param)

key, _ = jax.random.split(key)
ys_clean_np = render_observations(xs_np, g_true_renderer, noise_scale=0.0)
ys_clean = jnp.array(ys_clean_np)
save_dir_comparison = 'experiments/pendulum/evaluation/'


def loss_fn(model_: NNPendulum_decoder_SSM, key_):
    def logpdf_y_cond_x_NN_(y, x):
        """
        Calculates log p(y | x) for image observations.
        - y: single observation, shape (img_height, img_width, 1)
        - x: state, shape (nparticles, 2)
        """
        training_noise = 0.2  # for training stability
        particle_means_pred = model_.g_observation(x)  # shape (nparticles, img_height, img_width, 1)

        # weighted likelihood for training efficiency
        pixel_is_background = (y < 0.1)
        loss_weights = jnp.where(pixel_is_background, 1.0, 100.0)  # shape (img_height, img_width, 1)

        # pixel-wise log likelihood, shape (nparticles, img_height, img_width, 1)
        log_probs_pixelwise = jax.scipy.stats.norm.logpdf(
            y,
            loc=particle_means_pred,
            scale=sigma_xi_pixel
        )

        # weighted_log_probs_pixelwise = log_probs_pixelwise * loss_weights  # shape (nparticles, img_height, img_width, 1)
        # return jnp.mean(weighted_log_probs_pixelwise, axis=(1, 2, 3))
        return jnp.mean(log_probs_pixelwise, axis=(1, 2, 3))

    def log_g0_NN_(samples, y0_):
        return logpdf_y_cond_x_NN_(y0_, samples)

    def m_log_g_NN_(key__, samples, y):
        qs_ = jax.random.normal(key__, shape=(nparticles, dx)) * sigma_q
        prop_samples = jax.vmap(model_.f_dynamics, in_axes=[0, 0])(samples, qs_)
        return logpdf_y_cond_x_NN_(y, prop_samples), prop_samples

    _, log_ws_, nll, *_ = smc_feynman_kac(
        key_,
        m0_sampler,
        log_g0_NN_,
        m_log_g_NN_,
        ys,
        nparticles,
        nsteps,
        resampling=resampling,
        resampling_threshold=resampling_threshold,
        return_path=False
    )
    return nll


@nnx.jit
def train_step(model_, optimiser_, key_):
    loss_, grads = nnx.value_and_grad(loss_fn)(model_, key_)
    optimiser_.update(model_, grads)  # TODO: double-check if this indeed updates
    return loss_


print_prefix = f'Diffres ({mc_id}) | a {a} | T {T} | dsteps={dsteps} | {integrator} {"| ode" if ode else "| sde"}'
filename_prefix = f'diffres-{a}-{T}-{dsteps}-{integrator}-{"ode" if ode else "sde"}-'
losses = np.zeros(args.niters)
for i in range(args.niters):
    key, _ = jax.random.split(key)
    loss = train_step(model, optimiser, key)
    losses[i] = loss
    print(print_prefix + f' | Iter {i} | nll {loss}')
    if i % 10 == 0:
        ys_pred = model.g_observation(xs)
        filename = filename_prefix + f'{mc_id}' + f'_decoder_comparison_iter_{i:04d}.png'
        save_comparison_grid(
            ys_clean,
            ys_pred,
            nsteps,
            save_dir=save_dir_comparison,
            filename=filename
        )
nnx_save(model, 'experiments/pendulum/checkpoints/' + filename_prefix + f'{mc_id}')


def simulate_dynamics_only(key_, dynamics_func, x_init, num_steps, use_noise=True):
    qs = jnp.zeros((num_steps, dx))
    if use_noise:
        key_q, _ = jax.random.split(key_)
        qs = jax.random.normal(key_q, shape=(num_steps, dx)) * sigma_q

    def scan_body(carry_x, q_k):
        next_x = dynamics_func(carry_x, q_k)
        return next_x, next_x

    _, xs_ = jax.lax.scan(scan_body, x_init, qs)
    return jnp.concatenate([x_init[jnp.newaxis, ...], xs_], axis=0)


def pred_err_per_path(key_):
    xs_pred = simulate_dynamics_only(
        key_,
        model.f_dynamics,
        x0,
        nsteps,
        use_noise=True
    )
    return jnp.mean((xs_pred - xs) ** 2) ** 0.5


key, _ = jax.random.split(key)
keys = jax.random.split(key, num=args.npreds)
pred_err = jnp.mean(jax.vmap(pred_err_per_path)(keys))
print(print_prefix + f' | Prediction RMSE {pred_err}')
os.makedirs('experiments/pendulum/results', exist_ok=True)
save_path = 'experiments/pendulum/results/' + filename_prefix + f'{mc_id}.npz'
np.savez_compressed(save_path,
                    losses=losses, pred_err=pred_err, xs=np.array(xs), ys=np.array(ys), print_prefix=print_prefix,
                    filename_prefix=filename_prefix, dt=dt, nsteps=nsteps, nparticles=nparticles)
print("Saved to:", os.path.abspath(save_path))
print("Exists?", os.path.exists(save_path))
print("Files in dir:", os.listdir(os.path.dirname(save_path)))

plt.figure()
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('experiments/pendulum/results/' + filename_prefix + f'{mc_id}_pendulum_loss_vs_iter.png')

num_simulations = 10
simulate_with_noise = True
key, true_keys_key, learned_keys_key = jax.random.split(key, 3)
true_sim_keys = jax.random.split(true_keys_key, num_simulations)
learned_sim_keys = jax.random.split(learned_keys_key, num_simulations)

simulate_batch = jax.vmap(simulate_dynamics_only, in_axes=(0, None, None, None, None))
true_trajectories = simulate_batch(true_sim_keys, f, x0, nsteps, simulate_with_noise)
learned_trajectories = simulate_batch(learned_sim_keys, model.f_dynamics, x0, nsteps, simulate_with_noise)

plot_prefix = 'experiments/pendulum/results/' + filename_prefix + f'{mc_id}'
plt.figure(figsize=(8, 8))
for i in range(num_simulations):
    plt.plot(true_trajectories[i, :, 0], true_trajectories[i, :, 1], 'b-', alpha=0.3, linewidth=1,
             label='True f' if i == 0 else "")
for i in range(num_simulations):
    plt.plot(learned_trajectories[i, :, 0], learned_trajectories[i, :, 1], 'r--', alpha=0.5, linewidth=1,
             label='Learned f' if i == 0 else "")
plt.title(f'{num_simulations} Simulated Trajectories ({"Stochastic" if simulate_with_noise else "Deterministic"})')
plt.xlabel('Angle (rad)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)
plt.axis('equal')
phase_plot_path = plot_prefix + 'phase_plot.png'
plt.savefig(phase_plot_path)
print(f"Saved phase plot to: {phase_plot_path}")
plt.close()

plt.figure(figsize=(12, 8))
time_axis = np.arange(nsteps + 1) * dt
plt.subplot(2, 1, 1)
plt.plot(time_axis, true_trajectories[0, :, 0], 'b-', label='True f Angle')
plt.plot(time_axis, learned_trajectories[0, :, 0], 'r--', label='Learned f Angle')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(time_axis, true_trajectories[0, :, 1], 'b-', label='True f Velocity')
plt.plot(time_axis, learned_trajectories[0, :, 1], 'r--', label='Learned f Velocity')
plt.ylabel('Angular Velocity (rad/s)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)
plt.suptitle('Time Series Comparison (Example Trajectory)')
ts_plot_path = plot_prefix + 'timeseries_plot.png'
plt.savefig(ts_plot_path)
print(f"Saved time series plot to: {ts_plot_path}")
plt.close()

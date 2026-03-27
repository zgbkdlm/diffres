import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path

DATASET       = 'cloud'   # 'cloud' or 'temperature'
CMAP          = 'Blues'   # colormap for the images #Blues #'RdBu_r'
MASKED_COLOR  = 'black'   # color of masked pixels in visualization

DATASETS = {
    'cloud': [('total_cloud_cover_5.625deg', 'total_cloud_cover_2010_5.625deg.nc', 'tcc'),],
    'temperature': [('temperature_850_5.625deg', 'temperature_850hPa_2010_5.625deg.nc', 't'),],
}

MASK_RATIO_PRETRAIN  = 0.01  # fraction masked out during pre-training  (99% observed)
MASK_RATIO_FINETUNE  = 0.20  # fraction masked out during fine-tuning   (80% observed)
NOISE_STD            = 0.01
N_STEPS              = 257

script_dir = Path(__file__).parent
out_dir    = script_dir / 'observations' / DATASET
out_dir.mkdir(parents=True, exist_ok=True)

channels = []

for folder_name, filename, varname in DATASETS[DATASET]:
    ds   = xr.open_dataset(script_dir / folder_name / filename)
    data = ds[varname].values
    print(f'Loaded {varname}: {data.shape}')

    if data.ndim == 4:
        data = data[:, 0, :, :]

    data_min = data.min()
    data_max = data.max()
    data     = ((data - data_min) / (data_max - data_min)).astype(np.float32)

    channels.append(data)

# stack to (T, H, W, C)
sequence_full = np.stack(channels, axis=-1)
print(f'Full sequence shape: {sequence_full.shape}')

T, H, W, C = sequence_full.shape

# subsample to N_STEPS 
stride   = T // N_STEPS
sequence = sequence_full[::stride][:N_STEPS]  # (N_STEPS, H, W, C)

print(f'Stride: {stride} steps = {stride * 6} hours = {stride * 6 / 24:.1f} days between frames')
print(f'Sequence shape: {sequence.shape}')

# nested masks 
rng = np.random.default_rng(seed=42)

# pre-train mask: nearly everything observed
mask_pretrain = rng.random((H, W, C)) > MASK_RATIO_PRETRAIN

# fine-tune mask: subset of pre-train mask, more pixels masked out
extra_mask    = rng.random((H, W, C)) > (MASK_RATIO_FINETUNE - MASK_RATIO_PRETRAIN)
mask_finetune = mask_pretrain & extra_mask

print(f'Pre-train observed:  {mask_pretrain.sum()} / {H*W*C} ({100*mask_pretrain.mean():.1f}%)')
print(f'Fine-tune observed:  {mask_finetune.sum()} / {H*W*C} ({100*mask_finetune.mean():.1f}%)')

# generate flattened measurement vector
def generate_measurement(x, mask, noise_std=NOISE_STD, rng=None):
    """
    x:    (H, W, C)
    mask: (H, W, C) boolean
    returns y: (N,) flat vector of observed pixels + noise
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, noise_std, size=mask.sum()).astype(np.float32)
    return x[mask].flatten() + noise

# measurements for both masks
measurements_pretrain = np.stack([
    generate_measurement(sequence[t], mask_pretrain, rng=rng)
    for t in range(N_STEPS)
])  # (N_STEPS, N_pretrain)

measurements_finetune = np.stack([
    generate_measurement(sequence[t], mask_finetune, rng=rng)
    for t in range(N_STEPS)
])  # (N_STEPS, N_finetune)

print(f'Measurements pre-train shape:  {measurements_pretrain.shape}')
print(f'Measurements fine-tune shape:  {measurements_finetune.shape}')

# save results
np.save(out_dir / 'sequence.npy',              sequence)
np.save(out_dir / 'mask_pretrain.npy',         mask_pretrain)
np.save(out_dir / 'mask_finetune.npy',         mask_finetune)
np.save(out_dir / 'measurements_pretrain.npy', measurements_pretrain)
np.save(out_dir / 'measurements_finetune.npy', measurements_finetune)

print(f'Saved sequence:               {sequence.shape}')
print(f'Saved mask_pretrain:          {mask_pretrain.shape}')
print(f'Saved mask_finetune:          {mask_finetune.shape}')
print(f'Saved measurements_pretrain:  {measurements_pretrain.shape}')
print(f'Saved measurements_finetune:  {measurements_finetune.shape}')

# visualize
num_images = 16
indices    = np.linspace(0, N_STEPS - 1, num_images, dtype=int)
cols, rows = 4, num_images // 4

cmap_obj = plt.get_cmap(CMAP).copy()
cmap_obj.set_bad(color=MASKED_COLOR)

def measurement_to_image(y_k, mask):
    img = np.full(mask.shape, np.nan, dtype=np.float32)
    img[mask] = y_k
    return img

def save_grid(imgs, indices, suptitle, savepath):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    for i, ax in enumerate(axes.flat):
        img = imgs[indices[i], :, :, 0].copy()
        ax.imshow(img, cmap=cmap_obj, origin='lower', vmin=0, vmax=1)
        ax.set_title(f't={indices[i]}', fontsize=8)
        ax.axis('off')
    plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {savepath}')

def save_measurement_grid(measurements, mask, indices, suptitle, savepath):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    for i, ax in enumerate(axes.flat):
        img = measurement_to_image(measurements[indices[i]], mask)
        ax.imshow(img[:, :, 0], cmap=cmap_obj, origin='lower', vmin=0, vmax=1)
        ax.set_title(f't={indices[i]}', fontsize=8)
        ax.axis('off')
    plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {savepath}')

# ground truth
save_grid(sequence, indices,
          f'{DATASET} — ground truth (X_k)',
          out_dir / 'ground_truth_grid.png')

# pre-train measurements
save_measurement_grid(measurements_pretrain, mask_pretrain, indices,
                      f'{DATASET} — measurements pre-train (Y_k, {100*(1-MASK_RATIO_PRETRAIN):.0f}% observed)',
                      out_dir / 'measurement_grid_pretrain.png')

# fine-tune measurements
save_measurement_grid(measurements_finetune, mask_finetune, indices,
                      f'{DATASET} — measurements fine-tune (Y_k, {100*(1-MASK_RATIO_FINETUNE):.0f}% observed)',
                      out_dir / 'measurement_grid_finetune.png')
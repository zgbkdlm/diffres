"""
Plot the prediction errors and loss traces of the Lokta model.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 21})

# Global params
nparticles = 64
num_mcs = 20
niters = 1000
training_iters = np.arange(niters)


def check_nan(loss, pred_err):
    if np.isnan(loss) or np.isnan(pred_err) or pred_err > 1000.:
        return True
    else:
        return False


diff_methods = ['-1.0-1.0-4-euler-sde',
                '-1.0-2.0-16-euler-sde']
diff_labels = ['$T=1$\n $K=4$',
               '$T=2$\n $K=16$']
epss = [0.3, 1.0]
taus = [0.1, 0.3, 0.5]
alphas = [0.5, 0.7, 0.9]

# Plot err statistics
fig, ax = plt.subplots(figsize=(9, 6))
means = []
all_errs = []      # collect arrays for boxplot
box_positions = [] # x positions where data exists


def load_errs(filename_prefix):
    """Load and clean errors for a given prefix. Returns None if files missing."""
    if not os.path.isfile(filename_prefix + f'0.npz'):
        return None
    errs_preds = np.zeros(num_mcs)
    nan_flags = np.zeros(num_mcs).astype(bool)
    for mc_id in range(num_mcs):
        data = np.load(filename_prefix + f'{mc_id}.npz')
        errs_preds[mc_id] = data['pred_err']
        nan_flags[mc_id] = check_nan(data['losses'][-1], errs_preds[mc_id])
    return errs_preds[~nan_flags]


scatter_ind = -1

# Diffusion
for i, method in enumerate(diff_methods):
    scatter_ind += 1
    errs_preds = load_errs(f'./lokta/results/diffres-' + method + '-')
    if errs_preds is None:
        print(f'{method} not tested. Pass')
        continue
    means.append(np.mean(errs_preds))
    all_errs.append(errs_preds)
    box_positions.append(scatter_ind)
    ax.scatter(scatter_ind * np.ones(errs_preds.shape[0]), errs_preds,
               s=50, edgecolors='none', facecolors='black', alpha=.5, zorder=3)

# OT
for i, eps in enumerate(epss):
    scatter_ind += 1
    errs_preds = load_errs(f'./lokta/results/ot-{eps}-')
    if errs_preds is None:
        print(f'OT {eps} loss not tested. Pass')
        continue
    means.append(np.mean(errs_preds))
    all_errs.append(errs_preds)
    box_positions.append(scatter_ind)
    ax.scatter(scatter_ind * np.ones(errs_preds.shape[0]), errs_preds,
               s=50, edgecolors='none', facecolors='black', alpha=.5, zorder=3)

# Gumbel
for i, tau in enumerate(taus):
    scatter_ind += 1
    errs_preds = load_errs(f'./lokta/results/gumbel-{tau}-')
    if errs_preds is None:
        print(f'Gumbel {tau} loss not tested. Pass')
        continue
    means.append(np.mean(errs_preds))
    all_errs.append(errs_preds)
    box_positions.append(scatter_ind)
    ax.scatter(scatter_ind * np.ones(errs_preds.shape[0]), errs_preds,
               s=50, edgecolors='none', facecolors='black', alpha=.5, zorder=3)

# Soft
for i, alpha in enumerate(alphas):
    scatter_ind += 1
    errs_preds = load_errs(f'./lokta/results/soft-{alpha}-')
    if errs_preds is None:
        print(f'Soft {alpha} loss not tested. Pass')
        continue
    means.append(np.mean(errs_preds))
    all_errs.append(errs_preds)
    box_positions.append(scatter_ind)
    ax.scatter(scatter_ind * np.ones(errs_preds.shape[0]), errs_preds,
               s=50, edgecolors='none', facecolors='black', alpha=.5, zorder=3)

# Stopped
scatter_ind += 1
errs_preds = load_errs(f'./lokta/results/stopped-')
if errs_preds is None:
    print(f'stopped loss not tested. Pass')
else:
    means.append(np.mean(errs_preds))
    all_errs.append(errs_preds)
    box_positions.append(scatter_ind)
    ax.scatter(scatter_ind * np.ones(errs_preds.shape[0]), errs_preds,
               s=50, edgecolors='none', facecolors='black', alpha=.5, zorder=3)

# Box plot overlay
bp = ax.boxplot(all_errs, positions=box_positions, widths=0.5,
                showfliers=False, patch_artist=True, zorder=2,
                medianprops=dict(color='black', linewidth=2),
                boxprops=dict(facecolor='none', edgecolor='black', linewidth=2),
                whiskerprops=dict(color='black', linewidth=2),
                capprops=dict(color='black', linewidth=2))

# ax.plot(np.arange(len(means)), means, c='black', linewidth=3, label='Mean')
ax.set_yscale('log')
ax.grid(linestyle='--', alpha=0.3, which='both')

xticks = ([label for label in diff_labels]
          + [f'OT {eps}' for eps in epss]
          + [f'Gumbel {tau}' for tau in taus]
          + [f'Soft {alpha}' for alpha in alphas]
          + ['Stopped'])
ax.set_xticks(np.arange(len(diff_methods) + len(epss) + len(taus) + len(alphas) + 1))
ax.set_xticklabels(xticks, rotation=40, ha='center')
ax.set_ylabel('Prediction RMSE')

plt.tight_layout(pad=.1)
plt.savefig('lokta-rmse.pdf', transparent=True)
plt.show()

"""
Plot the loss curves
Different runs may have fairly different loss curves, thus used median
"""
fig, ax = plt.subplots()

# Diffusion
method = diff_methods[1]
lossess = np.zeros((num_mcs, niters))
nan_flags = np.zeros(num_mcs).astype(bool)

filename_prefix = f'./lokta/results/diffres-' + method + '-'
for mc_id in range(num_mcs):
    data = np.load(filename_prefix + f'{mc_id}.npz')
    lossess[mc_id] = data['losses']
    nan_flags[mc_id] = check_nan(data['losses'][-1], data['pred_err'])

lossess = lossess[~nan_flags]
ax.plot(training_iters[::10], np.median(lossess, axis=0)[::10],
        c='black', linewidth=2, alpha=1., label=f'Diffusion {diff_labels[0].replace("\n", ",")}')

# OT
eps = epss[0]
lossess = np.zeros((num_mcs, niters))
nan_flags = np.zeros(num_mcs).astype(bool)

filename_prefix = f'./lokta/results/ot-{eps}-'
for mc_id in range(num_mcs):
    data = np.load(filename_prefix + f'{mc_id}.npz')
    lossess[mc_id] = data['losses']
    nan_flags[mc_id] = check_nan(data['losses'][-1], data['pred_err'])

lossess = lossess[~nan_flags]
ax.plot(training_iters[::10], np.median(lossess, axis=0)[::10],
        c='black', linewidth=2, linestyle='--', alpha=1., label=f'OT {eps}')

# Gumbel
tau = taus[1]
lossess = np.zeros((num_mcs, niters))
nan_flags = np.zeros(num_mcs).astype(bool)

filename_prefix = f'./lokta/results/gumbel-{tau}-'
for mc_id in range(num_mcs):
    data = np.load(filename_prefix + f'{mc_id}.npz')
    lossess[mc_id] = data['losses']
    nan_flags[mc_id] = check_nan(data['losses'][-1], data['pred_err'])

lossess = lossess[~nan_flags]
ax.plot(training_iters[::10], np.median(lossess, axis=0)[::10],
        c='black', linewidth=2, marker='x', markevery=10, markersize=10, alpha=1., label=f'Gumbel {tau}')

# Soft
alpha = alphas[0]
lossess = np.zeros((num_mcs, niters))
nan_flags = np.zeros(num_mcs).astype(bool)

filename_prefix = f'./lokta/results/soft-{alpha}-'
for mc_id in range(num_mcs):
    data = np.load(filename_prefix + f'{mc_id}.npz')
    lossess[mc_id] = data['losses']
    nan_flags[mc_id] = check_nan(data['losses'][-1], data['pred_err'])

lossess = lossess[~nan_flags]
ax.plot(training_iters[::10], np.median(lossess, axis=0)[::10],
        c='black', linewidth=2, marker='*', markevery=10, markersize=10, alpha=1., label=f'Soft {alpha}')

# Stopped
lossess = np.zeros((num_mcs, niters))
nan_flags = np.zeros(num_mcs).astype(bool)

filename_prefix = f'./lokta/results/stopped-'
for mc_id in range(num_mcs):
    data = np.load(filename_prefix + f'{mc_id}.npz')
    lossess[mc_id] = data['losses']
    nan_flags[mc_id] = check_nan(data['losses'][-1], data['pred_err'])

lossess = lossess[~nan_flags]
ax.plot(training_iters[::10], np.median(lossess, axis=0)[::10],
        c='black', linewidth=2, marker='s', markevery=10, markersize=10, alpha=1., label=f'Stopped')

ax.set_yscale('log')
ax.grid(linestyle='--', alpha=0.3, which='both')
ax.set_xlabel('Training iteration')
ax.set_ylabel('MLE loss')

plt.legend(fontsize=20)
plt.tight_layout(pad=.1)
plt.savefig('lokta-loss.pdf', transparent=True)
plt.show()

# Plot pred-vs-truth
# see another script.

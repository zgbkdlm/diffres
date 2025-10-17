import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 21})

style = '2d'

platform = 'gpu'
nsampless = [128, 256, 512, 1024, 2048, 4096, 8192]
nstepss = [4, 8, 16, 32]
epss = [0.1, 0.2, 0.4, 0.8]

container_diff = np.zeros((len(nsampless), len(nstepss)))
container_ot = np.zeros((len(nsampless), len(epss)))

for i, nsamples in enumerate(nsampless):
    for j, nsteps in enumerate(nstepss):
        data = np.load(f'./profiling_time/results/times-diffusion-{nsamples}-{nsteps}-{platform}.npy')
        container_diff[i, j] = np.mean(data)

for i, nsamples in enumerate(nsampless):
    for j, eps in enumerate(epss):
        data = np.load(f'./profiling_time/results/times-ot-{nsamples}-{eps}-{platform}.npy')
        container_ot[i, j] = np.mean(data)

if style == '3d':
    ax = plt.figure().add_subplot(projection='3d')

    for i, nsteps in enumerate(nstepss):
        ax.plot(np.log2(nsampless), np.log2(nsteps * np.ones(len(nsampless))), container_diff[:, i], c='black')

    for i, eps in enumerate(epss):
        ax.plot(np.log2(nsampless), np.log2(eps * 40 * np.ones(len(nsampless))), container_ot[:, i], c='tab:blue')
    ax.set_xlabel('Number of samples (log base 2)')
    ax.set_ylabel(r'Parameters (log base 2)')
    ax.set_zlabel('Average time (s)')
    plt.show()

else:
    fig, axes = plt.subplots(figsize=(11, 6), ncols=2)
    axes[0].plot(nsampless, container_diff[:, 0], c='black', linewidth=2, label='Diffusion')
    axes[0].plot(nsampless, container_ot[:, 0], c='black', linewidth=2, linestyle='--', label='OT')
    axes[0].set_yscale('log', base=10)
    axes[0].set_xscale('log', base=2)
    axes[0].set_xlabel('Number of samples')
    axes[0].set_ylabel('Average time (s)')

    axes[1].plot(nstepss, container_diff[-1, :], c='black', linewidth=2, label='Diffusion')
    ax2 = axes[1].twiny()
    ax2.plot(epss, container_ot[-1, :], c='black', linewidth=2, linestyle='--', label='OT')
    ax2.set_xscale('log', base=2)
    axes[1].set_xscale('log', base=2)
    ax2.set_yscale('log', base=10)
    axes[1].set_yscale('log', base=10)
    axes[1].set_xlabel(r'Number of time steps (bottom) and OT $\epsilon$ (top)')

    for ax in axes:
        ax.grid(linestyle='--', alpha=0.3, which='both')
    axes[0].legend(fontsize=21)

    plt.tight_layout(pad=.1)
    plt.subplots_adjust(top=.931, left=.082, right=.962)
    plt.show()

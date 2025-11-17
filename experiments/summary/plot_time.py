"""
Visualise the computational costs of diffusion vs OT.
"""
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

times_diff = np.zeros((len(nsampless), len(nstepss)))
times_ot = np.zeros((len(nsampless), len(epss)))

for i, nsamples in enumerate(nsampless):
    for j, nsteps in enumerate(nstepss):
        data = np.load(f'./profiling_time/results/times-diffusion-{nsamples}-{nsteps}-{platform}.npz')
        times_diff[i, j] = np.mean(data['times'])
        errs_diff = np.mean(data['errs'] ** 2, axis=-1) ** 0.5
        print(f'Diff nsteps={nsteps} nsamples={nsamples} err={np.mean(errs_diff)} std={np.std(errs_diff)}')

for i, nsamples in enumerate(nsampless):
    for j, eps in enumerate(epss):
        data = np.load(f'./profiling_time/results/times-ot-{nsamples}-{eps}-{platform}.npz')
        times_ot[i, j] = np.mean(data['times'])
        errs_ot = np.mean(data['errs'] ** 2, axis=-1) ** 0.5
        print(f'OT eps={eps} nsamples={nsamples} err={np.mean(errs_ot)} std={np.std(errs_ot)}')

if style == '3d':
    ax = plt.figure().add_subplot(projection='3d')

    for i, nsteps in enumerate(nstepss):
        ax.plot(np.log2(nsampless), np.log2(nsteps * np.ones(len(nsampless))), times_diff[:, i], c='black')

    for i, eps in enumerate(epss):
        ax.plot(np.log2(nsampless), np.log2(eps * 40 * np.ones(len(nsampless))), times_ot[:, i], c='tab:blue')
    ax.set_xlabel('Number of samples (log base 2)')
    ax.set_ylabel(r'Parameters (log base 2)')
    ax.set_zlabel('Average time (s)')
    plt.show()

else:
    fig, axes = plt.subplots(figsize=(11, 6), ncols=2, sharey='row')
    axes[0].plot(nsampless, times_diff[:, 0], c='black', linewidth=3, marker='x', markersize=15,
                 label=f'Diffusion ($K={nstepss[0]}$)')
    axes[0].plot(nsampless, times_diff[:, -1], c='black', linewidth=3,
                 label=f'Diffusion ($K={nstepss[-1]}$)')
    axes[0].plot(nsampless, times_ot[:, 0], c='black', linewidth=3, linestyle='--',
                 label=rf'OT ($\varepsilon={epss[0]}$)')
    axes[0].plot(nsampless, times_ot[:, 3], c='black', linewidth=3, linestyle='--', marker='x', markersize=15,
                 label=rf'OT ($\varepsilon={epss[3]}$)')
    axes[0].set_yscale('log', base=10)
    axes[0].set_xscale('log', base=2)
    axes[0].set_xlabel('Number of samples $N$')
    axes[0].set_ylabel('Average time (s)')

    axes[1].plot(nstepss, times_diff[-1, :], c='black', linewidth=3, label=f'Diffusion ($N={nsampless[-1]}$)')
    ax2 = axes[1].twiny()
    ax2.plot(epss, times_ot[-1, :], c='black', linewidth=3, linestyle='--', label=f'OT ($N={nsampless[-1]}$)')
    ax2.set_xscale('log', base=2)
    axes[1].set_xscale('log', base=2)
    ax2.set_yscale('log', base=10)
    axes[1].set_yscale('log', base=10)
    axes[1].set_xlabel(r'Number of time steps (bottom) and OT $\varepsilon$ (top)')

    for ax in axes:
        ax.grid(linestyle='--', alpha=0.3, which='both')
    axes[0].legend(fontsize=21)
    axes[1].legend(fontsize=21, loc='upper right')
    ax2.legend(fontsize=21, loc='lower right')

    plt.tight_layout(pad=.1)
    plt.subplots_adjust(top=.931, left=.086, right=.967)
    plt.savefig('time-comparison.pdf', transparent=True)
    plt.show()

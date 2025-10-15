import itertools
import os
import numpy as np


# Print for diffusion
a = [-0.5, ]
Ts = [3., 4., 5.]
nstepss = [8, 32, 128]
integrators = ['euler', 'lord_and_rougemont', 'jentzen_and_kloeden', 'tweedie']
types = ['ode', 'sde']
num_mcs = 100

for comb in list(itertools.product(a, Ts, nstepss, integrators, types)):
    a, T, nsteps, integrator, type = comb
    errs = np.zeros(num_mcs)
    filename_prefix = f'./gms/results/diffres-{a}-{T}-{nsteps}-{integrator}-{type}-'
    if os.path.isfile(filename_prefix + '0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{mc_id}.npz')
            errs[mc_id] = data['err']
        print(f'Diffres {a}-{T}-{nsteps}-{integrator}-{type} | err {np.mean(errs)}.')
    else:
        print(f'Diffres {a}-{T}-{nsteps}-{integrator}-{type} not tested. Pass')
        pass

# Print for OT
epss = [0.3, 0.6, 0.8, 0.9]
for eps in epss:
    errs = np.zeros(num_mcs)
    filename_prefix = f'./gms/results/ot-{eps}-'
    if os.path.isfile(filename_prefix + '0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{mc_id}.npz')
            errs[mc_id] = data['err']
        print(f'OT {eps} | err {np.mean(errs)}.')
    else:
        print(f'OT {eps} not tested. Pass')
        pass

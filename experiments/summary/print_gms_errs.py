"""
Tabulate the Gaussian mixture errors.
"""
import itertools
import os
import numpy as np
from diffres.tools import statistics2latex

# Print for diffusion
a = [-1., -1.5]
Ts = [1., 2., 3.]
nstepss = [8, 32, 128]
integrators = ['euler', 'lord_and_rougemont', 'jentzen_and_kloeden', 'tweedie']
types = ['ode', 'sde']
num_mcs = 100
scale_swd = -1
scale_var = -2
print(f'The SWD results are scaled by 10^({scale_swd}), and resampling variance by 10^({scale_var})')

for comb in list(itertools.product(a, Ts, nstepss, integrators, types)):
    a, T, nsteps, integrator, type = comb
    errs = np.zeros(num_mcs)
    residuals = np.zeros(num_mcs)
    filename_prefix = f'./gms/results/diffres-{a}-{T}-{nsteps}-{integrator}-{type}-'
    if os.path.isfile(filename_prefix + '0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{mc_id}.npz')
            errs[mc_id] = data['err']
            residuals[mc_id] = np.sum(data['residual'] ** 2)
        print(
            f'Diffres {a}-{T}-{nsteps}-{integrator}-{type} '
            f'| {statistics2latex(np.mean(errs), np.std(errs), scale_swd)} '
            f'| {statistics2latex(np.mean(residuals), np.std(residuals), scale_var)}.')
    else:
        print(f'Diffres {a}-{T}-{nsteps}-{integrator}-{type} not tested. Pass')
        pass

# Print for OT
epss = [0.3, 0.6, 0.8, 0.9]
for eps in epss:
    errs = np.zeros(num_mcs)
    residuals = np.zeros(num_mcs)
    filename_prefix = f'./gms/results/ot-{eps}-'
    if os.path.isfile(filename_prefix + '0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{mc_id}.npz')
            errs[mc_id] = data['err']
            residuals[mc_id] = np.sum(data['residual'] ** 2)
        print(f'OT {eps} '
              f'| {statistics2latex(np.mean(errs), np.std(errs), scale_swd)} '
              f'| {statistics2latex(np.mean(residuals), np.std(residuals), scale_var)}.')
    else:
        print(f'OT {eps} not tested. Pass')
        pass

# Print for Gumbel
taus = [0.1, 0.2, 0.4, 0.8]
for tau in taus:
    errs = np.zeros(num_mcs)
    residuals = np.zeros(num_mcs)
    filename_prefix = f'./gms/results/gumbel-{tau}-'
    if os.path.isfile(filename_prefix + '0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{mc_id}.npz')
            errs[mc_id] = data['err']
            residuals[mc_id] = np.sum(data['residual'] ** 2)
        print(
            f'Gumbel {tau} '
            f'| {statistics2latex(np.mean(errs), np.std(errs), scale_swd)} '
            f'| {statistics2latex(np.mean(residuals), np.std(residuals), scale_var)}.')
    else:
        print(f'Gumbel {tau} not tested. Pass')
        pass

# Print for Soft
alphas = [0., 0.2, 0.4, 0.8, 0.9]
for alpha in alphas:
    errs = np.zeros(num_mcs)
    residuals = np.zeros(num_mcs)
    filename_prefix = f'./gms/results/soft-{alpha}-'
    if os.path.isfile(filename_prefix + '0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{mc_id}.npz')
            errs[mc_id] = data['err']
            residuals[mc_id] = np.sum(data['residual'] ** 2)
        print(
            f'Soft {alpha} '
            f'| {statistics2latex(np.mean(errs), np.std(errs), scale_swd)} '
            f'| {statistics2latex(np.mean(residuals), np.std(residuals), scale_var)}.')
    else:
        print(f'Soft {alpha} not tested. Pass')
        pass

# Print for multinomial
errs = np.zeros(num_mcs)
residuals = np.zeros(num_mcs)
filename_prefix = f'./gms/results/multinomial-'
if os.path.isfile(filename_prefix + '0.npz'):
    for mc_id in range(num_mcs):
        data = np.load(filename_prefix + f'{mc_id}.npz')
        errs[mc_id] = data['err']
        residuals[mc_id] = np.sum(data['residual'] ** 2)
    print(
        f'Multinomial '
        f'| {statistics2latex(np.mean(errs), np.std(errs), scale_swd)} '
        f'| {statistics2latex(np.mean(residuals), np.std(residuals), scale_var)}.')
else:
    print(f'Multinomial not tested. Pass')
    pass

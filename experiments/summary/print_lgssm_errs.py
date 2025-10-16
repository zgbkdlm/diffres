import itertools
import os
import numpy as np

# Global params
nparticles = 32
num_mcs = 100

# Print for diffusion
a = [-0.5, ]
Ts = [1., 2., 3.]
dstepss = [4, 8, 16, 32]
integrators = ['euler', 'lord_and_rougemont', 'jentzen_and_kloeden', 'tweedie']
types = ['ode', 'sde']

for comb in list(itertools.product(a, Ts, dstepss, integrators, types)):
    a, T, dsteps, integrator, type = comb
    errs_loss = np.zeros(num_mcs)
    errs_kl = np.zeros(num_mcs)
    errs_bures = np.zeros(num_mcs)
    filename_prefix = f'./lgssm/results/diffres-{a}-{T}-{dsteps}-{integrator}-{type}-'
    if os.path.isfile(filename_prefix + f'{nparticles}-0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{nparticles}-{mc_id}.npz')
            errs_loss[mc_id] = data['err_loss']
            errs_kl[mc_id] = data['err_filtering_kl']
            errs_bures[mc_id] = data['err_filtering_bures']
        print(
            f'Diffres {a}-{T}-{dsteps}-{integrator}-{type} | loss err {np.mean(errs_loss)} | kl err {np.mean(errs_kl)} | bures err {np.mean(errs_bures)}.')
    else:
        print(f'Diffres {a}-{T}-{dsteps}-{integrator}-{type} not tested. Pass')
        pass

# Print for OT
epss = [0.4, 0.8, 1.6]
for eps in epss:
    errs_loss = np.zeros(num_mcs)
    errs_kl = np.zeros(num_mcs)
    errs_bures = np.zeros(num_mcs)
    filename_prefix = f'./lgssm/results/ot-{eps}-'
    if os.path.isfile(filename_prefix + f'{nparticles}-0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{nparticles}-{mc_id}.npz')
            errs_loss[mc_id] = data['err_loss']
            errs_kl[mc_id] = data['err_filtering_kl']
            errs_bures[mc_id] = data['err_filtering_bures']
        print(
            f'OT {eps} | loss err {np.mean(errs_loss)} | kl err {np.mean(errs_kl)} | bures err {np.mean(errs_bures)}.')
    else:
        print(f'OT {eps} not tested. Pass')
        pass

# Print for Gumbel
taus = [0.2, 0.4, 0.8]
for tau in taus:
    errs_loss = np.zeros(num_mcs)
    errs_kl = np.zeros(num_mcs)
    errs_bures = np.zeros(num_mcs)
    filename_prefix = f'./lgssm/results/gumbel-{tau}-'
    if os.path.isfile(filename_prefix + f'{nparticles}-0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{nparticles}-{mc_id}.npz')
            errs_loss[mc_id] = data['err_loss']
            errs_kl[mc_id] = data['err_filtering_kl']
            errs_bures[mc_id] = data['err_filtering_bures']
        print(
            f'Gumbel {tau} | loss err {np.mean(errs_loss)} | kl err {np.mean(errs_kl)} | bures err {np.mean(errs_bures)}.')
    else:
        print(f'Gumbel {tau} not tested. Pass')
        pass

# Print for Soft
alphas = [0.5, 0.7, 0.9]
for alpha in alphas:
    errs_loss = np.zeros(num_mcs)
    errs_kl = np.zeros(num_mcs)
    errs_bures = np.zeros(num_mcs)
    filename_prefix = f'./lgssm/results/soft-{alpha}-'
    if os.path.isfile(filename_prefix + f'{nparticles}-0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{nparticles}-{mc_id}.npz')
            errs_loss[mc_id] = data['err_loss']
            errs_kl[mc_id] = data['err_filtering_kl']
            errs_bures[mc_id] = data['err_filtering_bures']
        print(
            f'Soft {alpha} | loss err {np.mean(errs_loss)} | kl err {np.mean(errs_kl)} | bures err {np.mean(errs_bures)}.')
    else:
        print(f'Soft {alpha} not tested. Pass')
        pass

# Print for multinomial
errs_loss = np.zeros(num_mcs)
errs_kl = np.zeros(num_mcs)
errs_bures = np.zeros(num_mcs)
filename_prefix = f'./lgssm/results/multinomial-'
if os.path.isfile(filename_prefix + f'{nparticles}-0.npz'):
    for mc_id in range(num_mcs):
        data = np.load(filename_prefix + f'{nparticles}-{mc_id}.npz')
        errs_loss[mc_id] = data['err_loss']
        errs_kl[mc_id] = data['err_filtering_kl']
        errs_bures[mc_id] = data['err_filtering_bures']
    print(
        f'Multinomial | loss err {np.mean(errs_loss)} | kl err {np.mean(errs_kl)} | bures err {np.mean(errs_bures)}.')
else:
    print(f'Multinomial not tested. Pass')
    pass

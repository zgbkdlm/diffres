"""
Tabulate the errors on the LGSSM.
"""
import itertools
import os
import numpy as np

# Global params
nparticles = 32
num_mcs = 100
true_params = np.array([0.5, 1.])


def check_success(errs_param, lbfgs_flat):
    if lbfgs_flat and errs_param < 1.9:
        return True
    else:
        return False


# Print for diffusion
a = [-1., ]
Ts = [1., 2., 3.]
dstepss = [4, 8, 16, 32]
integrators = ['euler', 'lord_and_rougemont', 'jentzen_and_kloeden', 'tweedie']
types = ['ode', 'sde']

for comb in list(itertools.product(a, Ts, dstepss, integrators, types)):
    a, T, dsteps, integrator, type = comb
    errs_loss = np.zeros(num_mcs)
    errs_kl = np.zeros(num_mcs)
    errs_bures = np.zeros(num_mcs)
    errs_params = np.zeros(num_mcs)
    successes = np.zeros(num_mcs).astype(bool)

    filename_prefix = f'./lgssm/results/diffres-{a}-{T}-{dsteps}-{integrator}-{type}-'
    if os.path.isfile(filename_prefix + f'{nparticles}-0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{nparticles}-{mc_id}.npz')
            errs_loss[mc_id] = data['err_loss'] ** 0.5
            errs_kl[mc_id] = data['err_filtering_kl']
            errs_bures[mc_id] = data['err_filtering_bures']
            errs_params[mc_id] = np.sum((true_params - data['opt_params']) ** 2) ** 0.5
            successes[mc_id] = check_success(errs_params[mc_id], data['opt_success'])

        num_success = np.sum(successes)
        print(
            f'Diffres {a}-{T}-{dsteps}-{integrator}-{type} '
            f'| loss err {np.mean(errs_loss)} +- {np.std(errs_loss)}'
            f'| kl err {np.mean(errs_kl):.2e} +- {np.std(errs_kl):.2e}'
            f'| bures err {np.mean(errs_bures):.2e} +- {np.std(errs_bures):.2e}'
            f'| params err {np.mean(errs_params[successes]):.2e} +- {np.std(errs_params[successes]):.2e}'
            f'| Successes {num_success}')
    else:
        print(f'Diffres {a}-{T}-{dsteps}-{integrator}-{type} not tested. Pass')
        pass

# Print for OT
epss = [0.4, 0.8, 1.6]
for eps in epss:
    errs_loss = np.zeros(num_mcs)
    errs_kl = np.zeros(num_mcs)
    errs_bures = np.zeros(num_mcs)
    errs_params = np.zeros(num_mcs)
    successes = np.zeros(num_mcs).astype(bool)

    filename_prefix = f'./lgssm/results/ot-{eps}-'
    if os.path.isfile(filename_prefix + f'{nparticles}-0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{nparticles}-{mc_id}.npz')
            errs_loss[mc_id] = data['err_loss'] ** 0.5
            errs_kl[mc_id] = data['err_filtering_kl']
            errs_bures[mc_id] = data['err_filtering_bures']
            errs_params[mc_id] = np.sum((true_params - data['opt_params']) ** 2) ** 0.5
            successes[mc_id] = check_success(errs_params[mc_id], data['opt_success'])

        num_success = np.sum(successes)
        print(
            f'OT {eps} '
            f'| loss err {np.mean(errs_loss)} +- {np.std(errs_loss)}'
            f'| kl err {np.mean(errs_kl):.2e} +- {np.std(errs_kl):.2e}'
            f'| bures err {np.mean(errs_bures):.2e} +- {np.std(errs_bures):.2e}'
            f'| params err {np.mean(errs_params[successes]):.2e} +- {np.std(errs_params[successes]):.2e}'
            f'| Successes {num_success}')
    else:
        print(f'OT {eps} not tested. Pass')
        pass

# Print for Gumbel
taus = [0.1, 0.3, 0.5]
for tau in taus:
    errs_loss = np.zeros(num_mcs)
    errs_kl = np.zeros(num_mcs)
    errs_bures = np.zeros(num_mcs)
    errs_params = np.zeros(num_mcs)
    successes = np.zeros(num_mcs).astype(bool)

    filename_prefix = f'./lgssm/results/gumbel-{tau}-'
    if os.path.isfile(filename_prefix + f'{nparticles}-0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{nparticles}-{mc_id}.npz')
            errs_loss[mc_id] = data['err_loss'] ** 0.5
            errs_kl[mc_id] = data['err_filtering_kl']
            errs_bures[mc_id] = data['err_filtering_bures']
            errs_params[mc_id] = np.sum((true_params - data['opt_params']) ** 2) ** 0.5
            successes[mc_id] = check_success(errs_params[mc_id], data['opt_success'])

        num_success = np.sum(successes)
        print(
            f'Gumbel {tau} '
            f'| loss err {np.mean(errs_loss)} +- {np.std(errs_loss)}'
            f'| kl err {np.mean(errs_kl):.2e} +- {np.std(errs_kl):.2e}'
            f'| bures err {np.mean(errs_bures):.2e} +- {np.std(errs_bures):.2e}'
            f'| params err {np.mean(errs_params[successes]):.2e} +- {np.std(errs_params[successes]):.2e}'
            f'| Successes {num_success}')
    else:
        print(f'Gumbel {tau} not tested. Pass')
        pass

# Print for Soft
alphas = [0.5, 0.7, 0.9]
for alpha in alphas:
    errs_loss = np.zeros(num_mcs)
    errs_kl = np.zeros(num_mcs)
    errs_bures = np.zeros(num_mcs)
    errs_params = np.zeros(num_mcs)
    successes = np.zeros(num_mcs).astype(bool)

    filename_prefix = f'./lgssm/results/soft-{alpha}-'
    if os.path.isfile(filename_prefix + f'{nparticles}-0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{nparticles}-{mc_id}.npz')
            errs_loss[mc_id] = data['err_loss'] ** 0.5
            errs_kl[mc_id] = data['err_filtering_kl']
            errs_bures[mc_id] = data['err_filtering_bures']
            errs_params[mc_id] = np.sum((true_params - data['opt_params']) ** 2) ** 0.5
            successes[mc_id] = check_success(errs_params[mc_id], data['opt_success'])

        num_success = np.sum(successes)
        print(
            f'Soft {alpha} '
            f'| loss err {np.mean(errs_loss)} +- {np.std(errs_loss)}'
            f'| kl err {np.mean(errs_kl):.2e} +- {np.std(errs_kl):.2e}'
            f'| bures err {np.mean(errs_bures):.2e} +- {np.std(errs_bures):.2e}'
            f'| params err {np.mean(errs_params[successes]):.2e} +- {np.std(errs_params[successes]):.2e}'
            f'| Successes {num_success}')
    else:
        print(f'Soft {alpha} not tested. Pass')
        pass

# Print for multinomial
errs_loss = np.zeros(num_mcs)
errs_kl = np.zeros(num_mcs)
errs_bures = np.zeros(num_mcs)
errs_params = np.zeros(num_mcs)
successes = np.zeros(num_mcs).astype(bool)

filename_prefix = f'./lgssm/results/multinomial-'
if os.path.isfile(filename_prefix + f'{nparticles}-0.npz'):
    for mc_id in range(num_mcs):
        data = np.load(filename_prefix + f'{nparticles}-{mc_id}.npz')
        errs_loss[mc_id] = data['err_loss'] ** 0.5
        errs_kl[mc_id] = data['err_filtering_kl']
        errs_bures[mc_id] = data['err_filtering_bures']
        errs_params[mc_id] = np.sum((true_params - data['opt_params']) ** 2) ** 0.5
        successes[mc_id] = check_success(errs_params[mc_id], data['opt_success'])

    num_success = np.sum(successes)
    print(
        f'Multinomial '
        f'| loss err {np.mean(errs_loss)} +- {np.std(errs_loss)}'
        f'| kl err {np.mean(errs_kl):.2e} +- {np.std(errs_kl):.2e}'
        f'| bures err {np.mean(errs_bures):.2e} +- {np.std(errs_bures):.2e}'
        f'| params err {np.mean(errs_params[successes]):.2e} +- {np.std(errs_params[successes]):.2e}'
        f'| Successes {num_success}')
else:
    print(f'Multinomial not tested. Pass')
    pass

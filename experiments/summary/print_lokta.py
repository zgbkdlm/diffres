import os
import numpy as np
import itertools

# Global params
nparticles = 64
num_mcs = 20


def check_nan(loss, pred_err):
    if np.isnan(loss) or np.isnan(pred_err) or pred_err > 50.:
        return True
    else:
        return False


# Print for diffusion
a = [-1., ]
Ts = [1., 2.]
dstepss = [4, 8, 16]
integrators = ['euler', 'lord_and_rougemont', 'jentzen_and_kloeden', 'tweedie']
types = ['ode', 'sde']

for comb in list(itertools.product(a, Ts, dstepss, integrators, types)):
    a, T, dsteps, integrator, type = comb
    errs_preds = np.zeros(num_mcs)
    errs_loss = np.zeros(num_mcs)
    nan_flags = np.zeros(num_mcs).astype(bool)

    filename_prefix = f'./lokta/results/diffres-{a}-{T}-{dsteps}-{integrator}-{type}-'
    if os.path.isfile(filename_prefix + f'0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{mc_id}.npz')
            errs_preds[mc_id] = data['pred_err']
            # Note that the target nll is only a reference for checking convergence,
            # it will not be informative to compare based on it
            errs_loss[mc_id] = np.abs(data['losses'][-1] - data['target_nll'])
            nan_flags[mc_id] = check_nan(data['losses'][-1], errs_preds[mc_id])

        num_success = np.sum(~nan_flags)
        print(
            f'Diffres {a}-{T}-{dsteps}-{integrator}-{type} '
            f'| pred err {np.mean(errs_preds[~nan_flags]):.2e} '
            f'| Success {num_success}')
    else:
        print(f'Diffres {a}-{T}-{dsteps}-{integrator}-{type} not tested. Pass')
        pass

# Print for OT
epss = [0.5, 1.0, 1.5]

for eps in epss:
    errs_preds = np.zeros(num_mcs)
    errs_loss = np.zeros(num_mcs)
    nan_flags = np.zeros(num_mcs).astype(bool)

    filename_prefix = f'./lokta/results/ot-{eps}-'
    if os.path.isfile(filename_prefix + f'0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{mc_id}.npz')
            errs_preds[mc_id] = data['pred_err']
            errs_loss[mc_id] = np.abs(data['losses'][-1] - data['target_nll'])
            nan_flags[mc_id] = check_nan(data['losses'][-1], errs_preds[mc_id])

        num_success = np.sum(~nan_flags)
        print(
            f'OT {eps} '
            f'| pred err {np.mean(errs_preds[~nan_flags]):.2e} '
            f'| Success {num_success}')
    else:
        print(f'OT {eps} loss not tested. Pass')
        pass

# Print for Gumbel
taus = [0.1, 0.3, 0.5]

for tau in taus:
    errs_preds = np.zeros(num_mcs)
    errs_loss = np.zeros(num_mcs)
    nan_flags = np.zeros(num_mcs).astype(bool)

    filename_prefix = f'./lokta/results/gumbel-{tau}-'
    if os.path.isfile(filename_prefix + f'0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{mc_id}.npz')
            errs_preds[mc_id] = data['pred_err']
            errs_loss[mc_id] = np.abs(data['losses'][-1] - data['target_nll'])
            nan_flags[mc_id] = check_nan(data['losses'][-1], errs_preds[mc_id])

        num_success = np.sum(~nan_flags)
        print(
            f'gumbel {tau} '
            f'| pred err {np.mean(errs_preds[~nan_flags]):.2e} '
            f'| Success {num_success}')
    else:
        print(f'gumbel {tau} loss not tested. Pass')
        pass

# Print for Soft
alphas = [0.5, 0.7, 0.9]

for alpha in alphas:
    errs_preds = np.zeros(num_mcs)
    errs_loss = np.zeros(num_mcs)
    nan_flags = np.zeros(num_mcs).astype(bool)

    filename_prefix = f'./lokta/results/soft-{alpha}-'
    if os.path.isfile(filename_prefix + f'0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{mc_id}.npz')
            errs_preds[mc_id] = data['pred_err']
            errs_loss[mc_id] = np.abs(data['losses'][-1] - data['target_nll'])
            nan_flags[mc_id] = check_nan(data['losses'][-1], errs_preds[mc_id])

        num_success = np.sum(~nan_flags)
        print(
            f'soft {alpha} '
            f'| pred err {np.mean(errs_preds[~nan_flags]):.2e} '
            f'| Success {num_success}')
    else:
        print(f'soft {alpha} loss not tested. Pass')
        pass

import os
import numpy as np
import itertools

# Global params
nparticles = 64
num_mcs = 10

# Print for diffusion
a = [-0.5, ]
Ts = [1., 2.]
dstepss = [4, 8, 16]
integrators = ['euler', 'lord_and_rougemont', 'jentzen_and_kloeden', 'tweedie']
types = ['ode', 'sde']

for comb in list(itertools.product(a, Ts, dstepss, integrators, types)):
    a, T, dsteps, integrator, type = comb
    filename_prefix = f'./lokta/results/diffres-{a}-{T}-{dsteps}-{integrator}-{type}-'
    if os.path.isfile(filename_prefix + f'0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{mc_id}.npz')
            print(
                f'Diffres {a}-{T}-{dsteps}-{integrator}-{type} | loss {data["losses"][-1]} | target loss {data["target_nll"]} pred err {data["pred_err"]}.')
    else:
        print(f'Diffres {a}-{T}-{dsteps}-{integrator}-{type} not tested. Pass')
        pass

# Print for OT
epss = [0.5, 1.0, 1.5]

for eps in epss:
    filename_prefix = f'./lokta/results/ot-{eps}-'
    if os.path.isfile(filename_prefix + f'0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{mc_id}.npz')
            print(
                f'OT {eps} loss {data["losses"][-1]} | target loss {data["target_nll"]} pred err {data["pred_err"]}.')
    else:
        print(f'OT {eps} loss not tested. Pass')
        pass

# Print for Gumbel
taus = [0.2, 0.4, 0.6]

for tau in taus:
    filename_prefix = f'./lokta/results/gumbel-{tau}-'
    if os.path.isfile(filename_prefix + f'0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{mc_id}.npz')
            print(
                f'gumbel {tau} loss {data["losses"][-1]} | target loss {data["target_nll"]} pred err {data["pred_err"]}.')
    else:
        print(f'gumbel {tau} loss not tested. Pass')
        pass


# Print for Soft
alphas = [0.5, 0.7, 0.9]

for alpha in alphas:
    filename_prefix = f'./lokta/results/soft-{alpha}-'
    if os.path.isfile(filename_prefix + f'0.npz'):
        for mc_id in range(num_mcs):
            data = np.load(filename_prefix + f'{mc_id}.npz')
            print(
                f'soft {alpha} loss {data["losses"][-1]} | target loss {data["target_nll"]} pred err {data["pred_err"]}.')
    else:
        print(f'soft {alpha} loss not tested. Pass')
        pass

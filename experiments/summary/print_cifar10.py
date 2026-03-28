import argparse
import jax
import jax.numpy as jnp
import torch
import numpy as np
from torchmetrics.functional.classification import multiclass_calibration_error, multiclass_f1_score
from diffres.data import CIFAR10
from diffres.nns import pbnn_cifar10
from diffres.tools import accuracy

parser = argparse.ArgumentParser(description='CIFAR10 classification.')
parser.add_argument('--nparticles', type=int, default=8, help='The number of SMC samples.')
parser.add_argument('--depth', type=int, default=18, help='The resnet depth.')
args = parser.parse_args()

# Random key seed
# Separate the key branch for data and algorithm
key = np.load('rnd_keys.npy')[0]
data_key, key = jax.random.split(key)

# Dataset creation
dataset = CIFAR10(data_key)

# Define the pBNN
key, _ = jax.random.split(key)
pbnn_phi, pbnn_psi, pbnn_forward_pass = pbnn_cifar10(key, 2, depth=args.depth)
pbnn_forward_pass = jax.jit(pbnn_forward_pass)

# Configurations
methods = ['diffusion', 'gumbel', 'soft', 'baseline']
num_mcs = 5
nparticles = args.nparticles

for method in methods:
    accs = np.zeros((num_mcs, nparticles))
    eces = np.zeros((num_mcs, nparticles))
    f1s = np.zeros((num_mcs, nparticles))
    log_wss = np.zeros((num_mcs, nparticles))
    for mc_id in range(num_mcs):
        filename = f'./cifar10/checkpoints/cp-{method}-{mc_id}.npz'
        cp = np.load(filename)
        psi = cp['psi']
        log_ws = cp['log_ws']
        samples = cp['samples']

        # Compute accuracy and ECE
        log_wss[mc_id] = log_ws
        for i in range(nparticles):
            pred = pbnn_forward_pass(samples[i], psi, dataset.test_xs[:100])
            accs[mc_id, i] = accuracy(pred, jax.nn.one_hot(dataset.test_ys[:100, 0], 10))
            f1s[mc_id, i] = multiclass_f1_score(torch.tensor(np.asarray(pred)),
                                                torch.tensor(np.asarray(dataset.test_ys[:100, 0])),
                                                num_classes=10)
            eces[mc_id, i] = multiclass_calibration_error(torch.tensor(np.asarray(pred)),
                                                          torch.tensor(np.asarray(dataset.test_ys[:100, 0])),
                                                          num_classes=10, n_bins=15, norm='l1')

    print(f'{method} acc', np.mean(np.sum(np.exp(log_wss) * accs, axis=-1)))
    print(f'{method} f1', np.mean(np.sum(np.exp(log_wss) * f1s, axis=-1)))
    print(f'{method} ece', np.mean(np.sum(np.exp(log_wss) * eces, axis=-1)))
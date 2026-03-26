import jax
import jax.numpy as jnp
import numpy as np
import math
import mnists
from diffres.typings import Array, JArray, JKey
from datasets import load_dataset
from abc import ABCMeta
from typing import List, Tuple, Callable


class DataSet(metaclass=ABCMeta):
    n: int
    xs: Array
    ys: Array
    rnd_inds: List

    val_n: int
    val_xs: Array
    val_ys: Array

    test_n: int
    test_xs: Array
    test_ys: Array

    @staticmethod
    def reshape(x):
        if x.ndim == 0:
            return jnp.reshape(x, (1, 1))
        elif x.ndim == 1:
            return jnp.reshape(x, (-1, 1))
        else:
            return x

    def init_enumeration(self, key, batch_size: int):
        """Randomly split the data into `n / batch_size` chunks. If the divisor is not an integer, then use // which
        truncates the training data.
        """
        n_chunks = self.n // batch_size
        self.rnd_inds = jnp.array_split(jax.random.choice(key,
                                                          jnp.arange(batch_size * n_chunks), (batch_size * n_chunks,),
                                                          replace=False),
                                        n_chunks)

    def enumerate_subset(self, i: int):
        inds = self.rnd_inds[i]
        return self.reshape(self.xs[inds, :]), self.reshape(self.ys[inds, :])

    def enumerate_all_batches(self, key, batch_size: int):
        n_chunks = self.n // batch_size
        perm_inds = jax.random.permutation(key, self.n)
        perm_inds = perm_inds[:n_chunks * batch_size]
        return (self.xs[perm_inds].reshape(n_chunks, batch_size, -1),
                self.ys[perm_inds].reshape(n_chunks, batch_size, -1))


class OneDimGaussian(DataSet):
    n: int

    xi: float
    fs: Callable

    def __init__(self, key: JKey, n: int, xs: JArray = None, xi: float = 1.):
        self.n = n
        self.xi = xi

        # Training data
        if xs is None:
            xs = jnp.sort(jax.random.uniform(key, shape=(n, 1), minval=-6., maxval=6.), axis=0)
        xs = jnp.reshape(xs, (-1, 1))
        self.xs = xs
        self.fs = lambda u: u * jnp.sin(u * jnp.tanh(u))
        key, subkey = jax.random.split(key)
        self.ys = self.fs(xs) + math.sqrt(xi) * jax.random.normal(subkey, (n, 1))

        # Validation data
        key, subkey = jax.random.split(key)
        self.val_xs = jnp.sort(jax.random.uniform(subkey, shape=(n, 1), minval=-6., maxval=6.), axis=0)
        key, subkey = jax.random.split(key)
        self.val_ys = self.fs(self.val_xs) + math.sqrt(xi) * jax.random.normal(subkey, (n, 1))

        # Test data
        key, subkey = jax.random.split(key)
        self.test_xs = jnp.sort(jax.random.uniform(subkey, shape=(n, 1), minval=-6., maxval=6.), axis=0)
        key, subkey = jax.random.split(key)
        self.test_ys = self.fs(self.test_xs) + math.sqrt(xi) * jax.random.normal(subkey, (n, 1))


class MNIST(DataSet):

    def __init__(self, key: JKey, which: str):
        mnist = mnists.MNIST() if which == 'mnist' else mnists.FashionMNIST()

        xs = jnp.asarray(np.concatenate([mnist.train_images(), mnist.test_images()],
                                        axis=0).reshape(70000, 784)) / 255
        ys = jnp.asarray(np.concatenate([mnist.train_labels(), mnist.test_labels()],
                                        axis=0).astype('int').reshape(70000, 1))
        perm_inds = jax.random.permutation(key, 70000)
        xs = xs[perm_inds]
        ys = ys[perm_inds]

        self.n = 50000
        self.n_val = 10000
        self.n_test = 10000

        # Training data
        self.xs = xs[:self.n]
        self.ys = ys[:self.n]

        # Validation data
        self.val_xs = xs[self.n:self.n + self.n_val]
        self.val_ys = ys[self.n:self.n + self.n_val]

        # Test data
        self.test_xs = xs[self.n + self.n_val:]
        self.test_ys = ys[self.n + self.n_val:]


class CIFAR10(DataSet):

    def __init__(self, key: JKey):
        # The dataset is already well shuffled, don't perm
        ds = load_dataset('uoft-cs/cifar10')

        train, test = ds['train'], ds['test']
        train_imgs = jnp.asarray(train['img']).reshape(50000, 3072) / 255
        train_labels = jnp.asarray(train['label']).astype('int').reshape(50000, 1)

        test_imgs = jnp.asarray(test['img']).reshape(10000, 3072) / 255
        test_labels = jnp.asarray(test['label']).astype('int').reshape(10000, 1)

        self.n = 50000
        self.n_val = 1000
        self.n_test = 10000

        # Training data
        self.xs = train_imgs[:self.n]
        self.ys = train_labels[:self.n]

        # Validation data
        self.val_xs = test_imgs[:self.n_val]
        self.val_ys = test_labels[:self.n_val]

        # Test data
        self.test_xs = test_imgs[self.n_val:]
        self.test_ys = test_labels[self.n_val:]

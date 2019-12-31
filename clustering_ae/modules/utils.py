"""Helper functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"

from typing import Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import mnist


def load_dataset(
    dataset: str = "mnist",
    batch_size: int = 64,
    one_hot: bool = True,
    flatten: bool = False,
    as_supervised: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray, np.ndarray]:
    """
    Returns a dataset object.

    Parameters
    ----------
    dataset : str
        The dataset to load.
    batch_size : int
        The mini-batch size.
    one_hot : bool
        Whether to onehot-encode the labels or not.
    flatten : bool
        Whether to flatten the vector or not.

    Returns
    -------
    train_dataset : tf.data.Dataset
        The training dataset object.
    test_dataset : tf.data.Dataset
        The test dataset object.
    test_features : numpy.ndarray
        The test features in NumPy array.
    test_labels : numpy.ndarray
        The test labels in NumPy array.
    """
    if (dataset == "mnist") or (dataset == "MNIST"):
        (train_features, train_labels), (test_features, test_labels) = mnist.load_data()
        train_features = train_features.reshape(-1, 28, 28, 1)
        test_features = test_features.reshape(-1, 28, 28, 1)
    elif (dataset == "fashion_mnist") or (dataset == "FMNIST"):
        (
            (train_features, train_labels),
            (test_features, test_labels),
        ) = fashion_mnist.load_data()
        train_features = train_features.reshape(-1, 28, 28, 1)
        test_features = test_features.reshape(-1, 28, 28, 1)
    elif (dataset == "cifar10") or (dataset == "CIFAR10"):
        (
            (train_features, train_labels),
            (test_features, test_labels),
        ) = cifar10.load_data()
        train_features = train_features.reshape(-1, 32, 32, 3)
        test_features = test_features.reshape(-1, 32, 32, 3)

    train_features = train_features.astype("float32")
    train_features = train_features / 255.0

    test_features = test_features.astype("float32")
    test_features = test_features / 255.0

    if flatten:
        dim = tf.math.reduce_prod(train_features.shape[1:]).numpy()
        train_features = train_features.reshape(-1, dim)
        test_features = test_features.reshape(-1, dim)

    if one_hot:
        classes = len(np.unique(train_labels))
        train_labels = tf.one_hot(train_labels, classes)
        test_labels = tf.one_hot(test_labels, classes)

    if as_supervised:
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_features, train_labels)
        )
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(batch_size * 4)
        train_dataset = train_dataset.shuffle(train_features.shape[0])

        test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
        test_dataset = test_dataset.batch(batch_size // 4)
        test_dataset = test_dataset.prefetch(batch_size * 4)
    elif not as_supervised:
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_features, train_features)
        )
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(batch_size * 4)
        train_dataset = train_dataset.shuffle(train_features.shape[0])

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (test_features, test_features)
        )
        test_dataset = test_dataset.batch(batch_size // 4)
        test_dataset = test_dataset.prefetch(batch_size * 4)

    return train_dataset, test_dataset, test_features, test_labels



def encode(features: np.ndarray, components: int = 3) -> np.ndarray:
    """
    Returns a PCA-encoded feature tensor.

    Parameters
    ----------
    features : np.ndarray
        The feature tensor to encode.
    components : int
        The number of components to which the `features` will be reduced to.

    Returns
    -------
    enc_features : np.ndarray
        The PCA-encoded features with `components`-dimension.
    """
    pca = PCA(n_components=components)
    enc_features = pca.fit_transform(features)
    return enc_features


def plot(features: np.ndarray, labels: np.ndarray, dimension: str = "3d") -> None:
    """
    Displays a scatter plot of `features` with respect to their labels.

    Parameters
    ----------
    features : np.ndarray
        The features tensor.
    labels : np.ndarray
        The labels tensor.
    dimension : str
        The dimensions of `features`. Either `2d` or `3d`.
    """
    sns.set_style("darkgrid")
    figure = plt.figure(figsize=(12, 8))

    if dimension == "3d" or features.shape[-1] == 3:
        ax = figure.gca(projection="3d")
        ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels, marker="o")
        ax.set_xlabel("feature 0")
        ax.set_ylabel("feature 1")
        ax.set_zlabel("feature 2")
        plt.show()
    elif dimension == "2d" or features.shape[-1] == 2:
        plt.scatter(features[:, 0], features[:, 1], c=labels, marker="o")
        plt.xlabel("feature 0")
        plt.ylabel("feature 1")
        plt.show()

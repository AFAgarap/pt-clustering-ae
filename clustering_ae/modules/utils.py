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
import tensorflow_datasets as tfds


def load_tfds(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns a data set from `tfds`.

    For this experiment, the options are:
        1. mnist
        2. fashion_mnist
        3. emnist/letters
        4. cifar10
        5. svhn_cropped

    Parameters
    ----------
    name : str
        The name of the TensorFlow data set to load.

    Returns
    -------
    train_features : np.ndarray
        The train features.
    test_features : np.ndarray
        The test features.
    train_labels : np.ndarray
        The train labels.
    test_labels : np.ndarray
        The test labels.
    """
    tf.config.experimental.set_memory_growth(
        tf.config.experimental.list_physical_devices("GPU")[0], True
    )
    train_dataset = tfds.load(name=name, split=tfds.Split.TRAIN, batch_size=-1)
    train_dataset = tfds.as_numpy(train_dataset)

    train_features = train_dataset["image"]
    train_labels = train_dataset["label"]

    train_features = train_features.astype("float32")
    train_features = train_features.reshape(-1, np.prod(train_features.shape[1:]))
    train_features = train_features / 255.0

    test_dataset = tfds.load(name=name, split=tfds.Split.TEST, batch_size=-1)
    test_dataset = tfds.as_numpy(test_dataset)

    test_features = test_dataset["image"]
    test_labels = test_dataset["label"]

    test_features = test_features.astype("float32")
    test_features = test_features.reshape(-1, np.prod(test_features.shape[1:]))
    test_features = test_features / 255.0

    return train_features, test_features, train_labels, test_labels


def encode(
    train_features: np.ndarray,
    test_features: np.ndarray,
    components: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """
    Returns a PCA-encoded feature tensor.

    Parameters
    ----------
    train_features : np.ndarray
        The training feature tensor to encode.
    test_features : np.ndarray
        The test feature tensor to encode.
    components : int
        The number of components to which the `features` will be reduced to.
    seed : int
        The random seed value.

    Returns
    -------
    enc_train_features : np.ndarray
        The PCA-encoded training features with `components`-dimension.
    enc_test_features : np.ndarray
        The PCA-encoded test features with `components`-dimension.
    """
    pca = PCA(n_components=components, random_state=seed)
    enc_train_features = pca.fit_transform(train_features)
    enc_test_features = pca.transform(test_features)
    return enc_train_features, enc_test_features


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

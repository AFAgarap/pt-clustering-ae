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


def load_dataset(
    dataset: str = "mnist",
    batch_size: int = 64,
    flatten: bool = False,
    as_supervised: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray, np.ndarray]:
    """
    Returns a tuple of dataset objects.

    Parameters
    ----------
    dataset : str
        The dataset to load.
    batch_size : int
        The mini-batch size.
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
    train_dataset = tfds.load(name=dataset, split=tfds.Split.TRAIN, batch_size=-1)
    test_dataset = tfds.load(name=dataset, split=tfds.Split.TEST, batch_size=-1)

    train_dataset = tfds.as_numpy(train_dataset)
    test_dataset = tfds.as_numpy(test_dataset)

    train_features = train_dataset["image"]
    train_labels = train_dataset["label"]
    test_features = test_dataset["image"]
    test_labels = test_dataset["label"]

    train_features = train_features.astype("float32")
    train_features = train_features / 255.0
    features_shape = train_features.shape[1:]

    test_features = test_features.astype("float32")
    test_features = test_features / 255.0

    if flatten:
        features_shape = np.prod(features_shape)
        train_features = train_features.reshape(-1, features_shape)
        test_features = test_features.reshape(-1, features_shape)

    num_classes = len(np.unique(train_labels))
    train_labels = tf.one_hot(train_labels, num_classes)
    test_labels = tf.one_hot(test_labels, num_classes)

    if as_supervised:
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_features, train_labels)
        )
        test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_features, train_labels)
        )
        test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))

    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size=batch_size)
    train_dataset = train_dataset.shuffle(train_features.shape[0])

    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size=batch_size)

    return (train_dataset, test_dataset, test_features, test_labels)


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

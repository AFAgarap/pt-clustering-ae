# Improving k-Means Clustering Performance with Disentangled Internal Representations
# Copyright (C) 2020  Abien Fred Agarap
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Helper functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"

import json
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


def create_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    as_supervised: bool = True,
) -> tf.data.Dataset:
    """
    Returns a `tf.data.Dataset` object from a pair of
    `features` and `labels` or `features` alone.

    Parameters
    ----------
    features : np.ndarray
        The features matrix.
    labels : np.ndarray
        The labels matrix.
    batch_size : int
        The mini-batch size.
    as_supervised : bool
        Boolean whether to load the dataset as supervised or not.

    Returns
    -------
    dataset : tf.data.Dataset
        The dataset pipeline object, ready for model usage.
    """
    if as_supervised:
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((features, features))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(features.shape[1])
    return dataset


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

    if dimension == "3d":
        ax = figure.gca(projection="3d")
        ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels, marker="o")
        ax.set_xlabel("feature 0")
        ax.set_ylabel("feature 1")
        ax.set_zlabel("feature 2")
        plt.show()
    elif dimension == "2d":
        plt.scatter(features[:, 0], features[:, 1], c=labels, marker="o")
        plt.xlabel("feature 0")
        plt.ylabel("feature 1")
        plt.show()


def clustering_accuracy(y_true, y_pred):
    """
    Metric from Guo et al., 2018
    [http://proceedings.mlr.press/v95/guo18b/guo18b.pdf]

    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    from sklearn.utils.linear_assignment_ import linear_assignment

    ind = linear_assignment(w.max() - w)

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def export_benchmark(results: dict, filename: str):
    """
    Exports the benchmark results.

    Parameters
    ----------
    results : dict
        The benchmark results.
    filename : str
        The filename of the exported results.
    """
    results = {key: str(value) for key, value in results.items()}
    with open("{}.json".format(filename), "w") as file:
        json.dump(results, file)


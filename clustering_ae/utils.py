# Improving k-Means Clustering Performance with Disentangled Internal Representations
# Copyright (C) 2020  Abien Fred Agarap
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Helper functions"""
import json
from typing import Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import linear_sum_assignment
import seaborn as sns
from sklearn.decomposition import PCA

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


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

    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)

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

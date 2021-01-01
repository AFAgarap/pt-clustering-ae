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
"""Implementation of k-Means Clustering with benchmarking"""
import time
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import silhouette_score

from clustering_ae.utils import clustering_accuracy


class Clustering(object):
    def __init__(
        self,
        num_clusters: int,
        n_init: int = 10,
        epochs: int = 300,
        seed: int = None,
        tol: float = 1e-4,
        initialization: str = "random",
    ):
        """
        k-Means Clustering

        Parameters
        ----------
        num_clusters : int
            The number of clusters to form.
        n_init : int, optional, default: 10
            The number of times k-Means will be run
            with varying centroid seeds.
        epochs : int, optional, default: 300
            The maximum number of iterations
            k-Means will be run.
        seed : int, optional, default: None
            The random number generator seed.
            Set for reproducibility.
        tol : float, optional, default: 1e-4
            The tolerance with regards to inertia.
        initialization : str, optional, default: random
            The method for initialization.
        """
        self.model = KMeans(
            init=initialization,
            n_clusters=num_clusters,
            n_init=n_init,
            max_iter=epochs,
            random_state=seed,
            tol=tol,
        )

    def __call__(self, features):
        """
        Wraps `self.predict`.

        Parameters
        ----------
        features: np.ndarray
            The instances to cluster.

        Returns
        -------
        np.ndarray
            The indices of the clusters.
        """
        return self.predict(features)

    def train(self, features):
        """
        Compute the clustering

        Parameter
        ---------
        features : np.ndarray
            The training instances to cluster.
        """
        self.model.fit(features)

    def predict(self, features):
        """
        Predict the clusters to which the features belongs to.

        Parameter
        ---------
        features : np.ndarray
            The test instances to cluster.
        """
        cluster_predictions = self.model.predict(features)
        return cluster_predictions

    def benchmark(self, name, features, labels):
        """
        Returns the clustering performance results in str and dict format.

        The metrics used are as follows:
            1. Duration
            2. Adjusted RAND Score
            3. Normalized Mutual Information
            4. Davies-Bouldin Index
            5. Silhouette Score
            6. Calinski-Harabasz Score
            7. Clustering Accuracy

        Parameters
        ----------
        name : str
            The name of the benchmark.
        features : np.ndarray
            The test instances to cluster.
        labels : np.ndarray
            The test labels.

        Returns
        -------
        str
            The formatted string of the benchmark results.
        results : dict
            The dictionary of benchmark results.
        """
        start_time = time.time()
        predictions = self.predict(features)

        results = {}

        results["name"] = name
        results["duration"] = time.time() - start_time
        results["ari"] = ari(labels_true=labels, labels_pred=predictions)
        results["nmi"] = nmi(labels_true=labels, labels_pred=predictions)
        results["dbi"] = davies_bouldin_score(features, predictions)
        results["silhouette"] = silhouette_score(
            features, predictions, metric="euclidean"
        )
        results["ch_score"] = calinski_harabasz_score(features, predictions)
        results["clustering_accuracy"] = clustering_accuracy(
            y_true=labels, y_pred=predictions
        )

        return (
            "%-9s\t%.2fs\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f"
            % (
                results["name"],
                results["duration"],
                results["dbi"],
                results["silhouette"],
                results["ch_score"],
                results["nmi"],
                results["ari"],
                results["clustering_accuracy"],
            ),
            results,
        )

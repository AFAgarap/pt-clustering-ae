# Using latent code from variational autoencoder for clustering
# Copyright (C) 2019  Abien Fred Agarap

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Implementation of k-Means Clustering in scikit-learn"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Abien Fred Agarap'

from sklearn.cluster import KMeans
from sklearn import metrics
from time import time


class Clustering():
    def __init__(self, num_clusters, n_init=10, epochs=300, cores=None):
        self.model = KMeans(init='k-means++', n_clusters=num_clusters, n_init=n_init, max_iter=epochs, n_jobs=cores)

    def train(self, training_data):
        self.model.fit(training_data)
    
    def predict(self, data):
        self.model.predict(data)

    def benchmark(self, name, data):
        start_time = time()
        predictions = self.model.predict(data)
        return ('%-9s\t%.2fs\t%.3f\t\t%.3f\t\t%.3f' % (name,
            (time() - start_time),
            metrics.davies_bouldin_score(data, predictions),
            metrics.silhouette_score(data, predictions, metric='euclidean'),
            metrics.calinski_harabaz_score(data, predictions)))

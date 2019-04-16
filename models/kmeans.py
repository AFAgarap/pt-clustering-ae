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

    def benchmark(self, name, data):
        start_time = time()
        predictions = self.model.predict(data)
        return ('%-9s\t%.2fs\t%.3f\t\t%.3f\t\t%.3f' % (name,
            (time() - start_time),
            metrics.davies_bouldin_score(data, predictions),
            metrics.silhouette_score(data, predictions, metric='euclidean'),
            metrics.calinski_harabaz_score(data, predictions)))

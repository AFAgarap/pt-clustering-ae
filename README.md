Improving k-Means Clustering Performance with Disentangled Internal Representations
===

by Abien Fred Agarap and Dr. Arnulfo P. Azcarraga

## Abstract

Deep clustering algorithms combine representation learning and clustering by jointly optimizing a clustering loss and a non-clustering loss. In such methods, a deep neural network is used for representation learning together with a clustering network. Instead of following this framework to improve clustering performance, we propose a simpler approach of optimizing the *entanglement* of the learned latent code representation of an autoencoder. We define *entanglement* as how close pairs of points from the same class or structure are, relative to pairs of points from different classes or structures. To measure the entanglement of data points, we use the *soft nearest neighbor loss*, and expand it by introducing an annealing temperature factor. Using our proposed approach, the test clustering accuracy was 96.2% on the MNIST dataset, 85.6% on the Fashion-MNIST dataset, and 79.2% on the EMNIST Balanced dataset, outperforming our baseline models.

## Learning Disentangled Representations

We consider the problem of clustering a set of *N* points {xi ∈ X} into *k* clusters, each represented by a centroid μj∈1,...,k. Instead of directly clustering the original features *X*, we transform the data with a non-linear mapping *Z = enc(X)*, where *Z* is the latent code representation. But to learn a more clustering-friendly representation, we propose to learn to *disentangle* them, i.e. isolate class- or structure-similar data points, which implicitly maximizes the inter-cluster variance.



### Clustering Performance

|   Method   |  ACC  |  NMI  |  ARI  |
| :--------: | :---: | :---: | :---: |
|    DEC     | 0.843 |   -   |   -   |
|    VaDE    | 0.945 |   -   |   -   |
|    N2D     | 0.948 | 0.882 |   -   |
| ClusterGAN | 0.95  | 0.89  | 0.89  |
|  AE+SNNL   | 0.962 | 0.903 | 0.918 |

**Table 1. Clustering Performance on the MNIST dataset.**



|   Method   |  ACC  |  NMI  |  ARI  |
| :--------: | :---: | :---: | :---: |
|    DEC     |   -   |   -   |   -   |
|    VaDE    |   -   |   -   |   -   |
|    N2D     | 0.672 | 0.684 |   -   |
| ClusterGAN | 0.63  | 0.64  |  0.5  |
|  AE+SNNL   | 0.856 | 0.767 | 0.729 |

**Table 2. Clustering Performance on the Fashion-MNIST dataset.**



|   Method   |  ACC  |  NMI  |  ARI  |
| :--------: | :---: | :---: | :---: |
|    DEC     |   -   |   -   |   -   |
|    VaDE    |   -   |   -   |   -   |
|    N2D     |   -   |   -   |   -   |
| ClusterGAN |   -   |   -   |   -   |
|  AE+SNNL   | 0.792 | 0.783 | 0.655 |

**Table 1. Clustering Performance on the EMNIST Balanced dataset.**





## License

[AGPL-3.0](LICENSE)

```
Improving k-Means Clustering Performance with Disentangled Internal Representations
Copyright (C) 2020  Abien Fred Agarap

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```

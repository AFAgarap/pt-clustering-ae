Improving k-Means Clustering Performance with Disentangled Internal Representations
===

by Abien Fred Agarap and Dr. Arnulfo P. Azcarraga

*This repository is undergoing revision*

## Abstract

Deep clustering algorithms combine representation learning and clustering by jointly optimizing a clustering and a non-clustering loss. In such methods, they use a deep neural network for representation learning, and connect it to a clustering network. Instead of following this framework for an improved clustering performance, we propose a simpler approach of minimizing the *entanglement* of the learned latent code representation of an autoencoder. We define
*entanglement* as how close pairs of points from the same class or structure are, relative to pairs of points from different classes or structures. To measure entanglement of data points, we use the *soft nearest neighbor loss*, and expand it by introducing an annlearning temperature factor. Using our proposed approach, we were able to achieve a test clustering accuracy of 96.2% on the MNIST dataset, 85.6% on the Fashion-MNIST dataset, and 79.2% on the EMNIST Balanced dataset, outperforming our baseline models.

## Learning Disentangled Representations

We consider the problem of clustering a set of *N* points {xi ∈ X} into *k* clusters, each represented by a centroid μj∈1,...,k. Instead of directly clustering the original features *X*, we transform the data with a non-linear mapping *Z = enc(X)*, where *Z* is the latent code representation. But in order to learn a more clustering-friendly representation, we propose to learn to *disentangle* them, i.e. isolate class- or structure-similar data points, which implicitly maximizes the inter-cluster variance.


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

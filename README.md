Improved Clustering Performance using Latent Data Representation from Variational Autoencoder
===

by Abien Fred Agarap and Arnulfo Azcarraga, PhD

## Abstract

Clustering is a well-studied task in terms of development of efficient algorithms, better initialization algorithms, and better distance functions. Conventionally, datasets are used as they are, or with some normalization done before using them as input to a clustering algorithm. In this study, we present using the learned latent data representation of a variational autoencoder as input to a clustering algorithm. We compared the clustering on the original data representation, on the reconstructed data, and on the learned latent data representation. Results have shown that the best clustering was on the latent data representation, with a Davies-Bouldin Index (DBI) of 1.328, Silhoutte Score (SS) of 0.238, and Calinski-Harabasz Score (CHS) of ~3609, while using the original data representation gained a DBI of 1.811, SS of 0.155, and CHS of ~1269, both on the Fashion-MNIST dataset. We had similar findings on the MNIST dataset as the best clustering was on the latent data representation as well, with a DBI of 1.557, SS of 0.186, and CHS of ~1497, while using the original data representation gained a DBI of 2.857, SS of 0.061, and CHS of ~391. This stands to reason since the latent data representation includes only the most salient features of a data, and its variance is minimized through the optimization of the VAE model.

## Results

### Experiment Setup
Experiments were done in Google Colaboratory that provides a free access to Tesla K80 GPU with a 12GB RAM allotment. We trained the VAE model with Adam optimization on the MNIST dataset (60,000 training images) for 60 epochs with mini-batches of 512 which took 3 minutes and 35 seconds, and on the Fashion-MNIST dataset (60,000 training images) for 60 epochs with mini-batches of 512 which took 3 minutes and 42 seconds.

For clustering, we trained the k-Means clustering algorithm using the 60,000 training images (from both MNIST and Fashion-MNIST) in its (1) original data representation, (2) latent code representation (from VAE), and (3) reconstructed data representation (from VAE). Then, we used the trained clustering model on the Gaussian noise-augmented 10,000 test images in all three data representations as we did in training.

### Visualization of Learned Representations

We visualize the reconstructed data by the VAE model on the MNIST and Fashion-MNIST datasets. Figure 1 shows the original data with Gaussian noise and a reconstructed sample from the trained VAE model on the MNIST dataset. Figure 2 shows the original data with Gaussian noise and a reconstructed sample from the trained VAE model on the Fashion-MNIST dataset. This finding corroborates the results from literature on the capability of VAE to denoise images

![](assets/mnist_noise_clean.png)
**Figure 1. The top row consists of the MNIST images with added Gaussian noise while the bottom row are the reconstructed images using the trained VAE.**

![](assets/fmnist_noise_clean.png)
**Figure 2. The top row consists of the Fashion-MNIST images with added Gaussian noise while the bottom row are the reconstructed images using the trained VAE.**

We then visualize the learned representation of the VAE model for the MNIST and Fashion-MNIST datasets. To this end, we used t-SNE visualization to reduce the dimensionality of the learned latent representation _z_ from 10 to 3, and the original data _x_ and the reconstructed data from 784 to 3. Figures 3 and 4 show the scatter plot of the features. We can see in these figures that the clusters in the learned latent data representations are denser compared to the clusters in the original data and in the reconstructed data.

![](assets/tsne_mnist.png)
**Figure 3. t-SNE visualization of the MNIST dataset with number of components = 3, and perplexity = 50. The left plot shows the t-SNE visualization for the original data, the middle for the reconstructed data, and the right for the latent code.**

![](assets/tsne_fmnist.png)
**Figure 4. t-SNE visualization of the Fashion-MNIST dataset with number of components = 3, and perplexity = 50. The left plot shows the t-SNE visualization for the original data, the middle for the reconstructed data, and the right for the latent code.**

### Clustering on Samples

We used a k-Means clustering model on the MNIST and Fashion-MNIST datasets with the *k-means++* initialization and with _k = 10_ since there are 10 classes in both MNIST and Fashion-MNIST, and trained it for 500 iterations per dataset input. The first input is the original data representation, the second is the reconstructed data by the VAE model, and the third is the latent code from the VAE model.

As we have seen in Figures 3 and 4, the clusters in the learned latent data representation are denser when compared to the clusters in the original data and in the reconstructed data. The clustering performance as shown in the Voronoi digrams (see Figure 5 and 6) is at its best when we use the latent data representation.

![](assets/clustering_mnist.png)
**Figure 5. Voronoi diagram for the k-Means clustering algorithm prediction on the MNIST dataset. The left diagram shows the clustering on the original dataset, the middle for the reconstructed data, and the right for the latent code.**

![](assets/clustering_fmnist.png)
**Figure 6. Voronoi diagram for the k-Means clustering algorithm prediction on the Fashion-MNIST dataset. The left diagram shows the clustering on the original dataset, the middle for the reconstructed data, and the right for the latent code.**

We posit the following reasons for this density: (1) only the most salient features are included in the latent space representation of the data - a simpler and lower-dimension data representation - this is also the reason behind its denoising capability, and (2) as per [12](https://arxiv.org/abs/1312.6114), [24](https://arxiv.org/abs/1401.0118), [27](https://projecteuclid.org/euclid.ba/1386166315), learning  the approximateinference inevitably results to lower variance, influencing thedata points to cluster near their expected value[9](https://www.deeplearningbook.org/) – which consequently helps the objective of the clustering algorithm, i.e. variance minimization.

Although these visual results do provide some evidence that the clustering performance may be improved through the use of latent code, we proceed to a more explicit measurement of the clustering quality as laid down in Tables 1 and 2.

**Table 1. Clustering Evaluation on the original data, reconstructed data, and latent code for the MNIST dataset.**
|Data|Davies-Bouldin Index|Silhouette Score|Calinski-Harabasz Score|
|----|--------------------|----------------|-----------------------|
|Original|2.857|0.061|391.245|
|Reconstructed|2.300|0.099|563.361|
|**Latent Code**|**1.557**|**0.186**|**1497.287**|

**Table 1. Clustering Evaluation on the original data, reconstructed data, and latent code for the Fashion-MNIST dataset.**
|Data|Davies-Bouldin Index|Silhouette Score|Calinski-Harabasz Score|
|----|--------------------|----------------|-----------------------|
|Original|1.811|0.155|1269.635|
|Reconstructed|1.419|0.216|2010.841|
|**Latent Code**|**1.328**|**0.238**|**3609.196**|

A low Davies-Bouldin Index denotes better cluster separations, while both high Silhouette Score and high Calinski-Harabasz Score denote better-defined clusters. As we can see, in both MNIST and Fashion-MNIST dataset, the clustering was at its best on the latent code from the VAE.
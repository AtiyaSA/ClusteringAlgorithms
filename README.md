# ClusteringAlgorithms
Implementation of Clustering Algorithms, namely kmeans, kmeans++ and bisecting kmeans clustering algorthims, from scratch.

<h2>Kmeans Clustering Algorithm</h2>
<br>
K-means is a clustering algorithm that aims to partition a set of data points into k clusters, where each point belongs to the cluster with the nearest mean (centroid). The algorithm starts by randomly selecting k initial centroids from the data points. Then, it iteratively performs two steps: (1) assignment and (2) update.<br>
In the assignment step, each data point is assigned to the nearest centroid based on Euclidean distance. This results in k clusters.<br>
In the update step, the centroids of the k clusters are recalculated based on the mean of the data points in each cluster. Then, the algorithm repeats the assignment and update steps until convergence.<br>
K-means is widely used due to its simplicity, efficiency, and effectiveness in many applications. However, it has some limitations such as sensitivity to the initial centroids, the assumption of spherical clusters, and the need to specify the number of clusters k in advance. Various extensions and modifications of k-means have been proposed to address these limitations, such as k-means++, hierarchical k-means, and fuzzy c-means.<br>
Here is the pseudo code for the k-means clustering algorithm:<br>
<dl>
<dt>1. Initialize k cluster centroids randomly</dt>
<dt>2. Repeat until convergence or maximum iterations:</dt>
<dd>a. Assign each data point to the nearest cluster centroid.</dd>
<dd>b. Update each cluster centroid to be the mean of all data points assigned to that cluster.</dd>
<dt>3. Return the final k clusters and their centroids</dt>
</dl><br>
In this implementation, X is the data matrix with each row representing a data point, k is the number of clusters, and max_iter is the maximum number of iterations to run the algorithm. The function returns the cluster labels for each data point and the final centroids for each cluster. The initial cluster representatives are selected randomly.<br>
The code loads the data into a pandas DataFrame, extracts the feature columns as a numpy array, and then calls the k_means function with k=5 and max_iter=100. Finally, it prints the cluster labels and the final centroids returned by the k_means function. Note that the initial cluster representatives are selected randomly by the k_means function.<br>

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

<h2>Kmeans++ Clustering Algorithm</h2>
<br>
k-Means++ is an algorithm used to initialize the centroids in the k-means clustering algorithm. The goal of the k-means++ algorithm is to select the initial centroids in a way that improves the chances of obtaining a better clustering result.<br>
The k-means++ algorithm works as follows:
<ol>
<li>1. The first centroid is chosen uniformly at random from the data points.</li>
<li>2. For each remaining centroid, a new candidate centroid is chosen with probability proportional to the square of the distance from the point to the closest existing centroid. This increases the probability of selecting points that are far away from existing centroids.</li>
<li>3. The process is repeated until all centroids have been initialized.</li>
</ol>
By selecting the initial centroids using k-means++, the algorithm is more likely to find a good solution, compared to choosing centroids randomly. The improved initialization can help prevent the algorithm from converging to a suboptimal solution and can reduce the number of iterations required for the algorithm to converge.<br>
Here is the pseudo code for the K-means++ algorithm:
<dl>
<dt>1. Choose the first centroid uniformly at random from the data points.</dt>
<dt>2. For i = 2 to k:</dl>
<dd>a. Compute the distance squared between each data point and the nearest centroid that has already been chosen.</dd>
<dd>b. Choose the next centroid from the remaining data points with probability proportional to the distance squared to the nearest existing centroid.</dd>
<dt>3. Initialize centroids to the k chosen points.</dt>
<dt>4. Perform K-means clustering on the data with the initial centroids from step 3.</dt>
<dt>5. Repeat step 4 until convergence or maximum number of iterations is reached.</dt>
<dt>6. Return the cluster assignments and final centroids.</dt>
<dl>
In the implementation, we first initialize the first cluster centre randomly by selecting a data point at random from the dataset. Then, we iterate k - 1 times, where k is the number of clusters we want to generate. In each iteration, we compute the squared distances to the nearest cluster centre for each data point and then select the next cluster centre from the data points with probability proportional to the squared distance. Finally, we run the standard k-means algorithm with the k initial cluster centres selected above.<br>

<h2>Bisecting Kmeans Clustering Algorithm</h2>
<br>
Bisecting k-Means is a hierarchical clustering algorithm that starts with all data points as a single cluster and recursively splits the clusters into two sub-clusters until the desired number of clusters is reached. At each iteration, the algorithm selects the cluster with the largest sum of squared distances (SSE) and applies the k-Means algorithm to bisect it into two new clusters.
The k-Means algorithm assigns each point in the selected cluster to the nearest of the two cluster centroids and iteratively updates the centroids until convergence. The two resulting sub-clusters are added to the cluster list, and the original cluster is removed. The algorithm repeats this process on the newly added clusters until the desired number of clusters is reached.
Unlike other hierarchical clustering algorithms, Bisecting k-Means always produces a binary tree of clusters, and it can be faster than traditional hierarchical clustering methods for large datasets. However, it can suffer from sensitivity to initial conditions and can produce unbalanced trees if the data is not well-suited for binary splitting.
Here is the pseudo code for the Bisecting k-Means algorithm:
<dl>
<dt>1. Initialize the cluster with all the data points.</dt>
<dt>2. While the desired number of clusters is not reached:</dt>
<dd>a. Select the largest cluster.</dd>
<dd>b. Apply the k-means algorithm to the selected cluster to obtain two sub-clusters</dd>
<dd>c. Remove the selected cluster and add the two sub-clusters to the cluster list</dd>
<dt>3. Return the list of clusters.</dt>
</dl>
In the implementation, we first initialize the current cluster to be the entire dataset and add it to the hierarchy list. Then, we iterate until the number of clusters in the hierarchy is equal to the desired number of clusters. In each iteration, we select the largest cluster in the hierarchy, bisect it into two smaller clusters using the standard k-means algorithm, and then compute the SSE of the resulting clusters. We then replace the selected cluster with the two resulting clusters in the hierarchy and continue the iteration. Finally, we return the list of labels for each data point based on the final hierarchy of clusters.

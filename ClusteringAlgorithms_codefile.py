#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

df = pd.read_csv('dataset', delim_whitespace=True,header = None)
df

df.describe()

def k_means(X, k, max_iter=100, centroids=None):
    # Initialize k random centroids
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for i in range(max_iter):
        # Assign each data point to the nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update the centroids to be the mean of the data points in each cluster
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        
        # Check if the centroids have moved
        if np.allclose(new_centroids, centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids


# Extract the feature columns as a numpy array
X = df.iloc[:, 1:].values

# Set the number of clusters and maximum iterations
k = 5
max_iter = 100

# Cluster the data using k-means
labels, centroids = k_means(X, k, max_iter)

def k_means_pp(X, k, max_iter):
    # Choose the first centroid uniformly at random
    centroids = [X[np.random.choice(X.shape[0])]]
    
    # Choose the remaining centroids using the K-means++ algorithm
    for i in range(k - 1):
        # Compute the distance squared to the nearest existing centroid for each point
        distances = np.array([min([np.linalg.norm(x-c)**2 for c in centroids]) for x in X])
        
        # Choose the next centroid with probability proportional to the distance squared
        probs = distances / distances.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        for j, p in enumerate(cumulative_probs):
            if r < p:
                next_centroid = X[j]
                break
        
        # Add the next centroid to the list of centroids
        centroids.append(next_centroid)
    
    # Cluster the data using K-means with the initial centroids
    labels, centroids = k_means(X, k, max_iter, centroids)
    
    return labels, centroids

# Extract the feature columns as a numpy array
X = df.iloc[:, 1:].values

# Set the number of clusters and maximum iterations
k = 5
max_iter = 100

# Cluster the data using k-means
labels, centroids = k_means_pp(X, k, max_iter)

# Print the cluster labels and the final centroids
print(len(labels))

def bisecting_kmeans(data, k, max_iterations=100):
    # Step 1: Initialize a single cluster C as all data points in D
    clusters = [data]
    while len(clusters) < k:
        # Step 2a: Select cluster with largest SSE
        max_sse,max_sse_idx = -1, -1
        for i in range(len(clusters)):
            sse = SSE(clusters[i])
            if sse > max_sse:
                max_sse = sse
                max_sse_idx = i
        # Select a cluster C in clusters that has the largest sum of square distance
        C = clusters[max_sse_idx]

        # Step 2b: Apply k-means algorithm to split largest SSE cluster
        labels, centers = k_means(C, 2)

        # Step 2c: Replace largest SSE cluster with two sub-clusters
        clusters.pop(max_sse_idx)
        clusters.append(C[labels == 0])
        clusters.append(C[labels == 1])

    # Return final clustering
    return clusters
def SSE(X):
    """Calculate the sum of squared errors (SSE) of a cluster."""
    centroid = np.mean(X, axis=0)
    return np.sum(np.square(X - centroid))
# Run the Bisecting k-Means algorithm on the dataset with k=3
clusters = bisecting_kmeans(X, k=3)

def silhouette_coefficient(data, cluster_assignments):
    distances = np.linalg.norm(data[:, None, :] - data[None, :, :], axis=2)
    a = np.zeros_like(cluster_assignments, dtype=np.float64)
    b = np.zeros_like(cluster_assignments, dtype=np.float64)
    for i in range(len(data)):
        cluster_i = cluster_assignments[i]
        a[i] = np.mean(distances[i, cluster_assignments == cluster_i])
        b[i] = np.min(np.mean(distances[i, cluster_assignments != cluster_i]))

    s = (b - a) / np.maximum(a, b)
    return np.mean(s)

# Question 4 - Compute the Silhouette coefficient for different values of k
silhouette_scores = []
print('k-means')
for k in range(1, 10):
    cluster_assignments, centroids = k_means(X, k, 100)
    score = silhouette_coefficient(X, cluster_assignments)
    print("For k =",k,", silhouette score = ", score)
    silhouette_scores.append(score)

# Plot the results

plt.plot(range(1, 10), silhouette_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette coefficient')
plt.title('Silhouette coefficient vs. k k-means')
plt.show()

# Question 5 - Compute the Silhouette coefficient for different values of k for k-means++
silhouette_scores = []
print('k-means++')
for k in range(1, 10):
    cluster_assgn, centroids = k_means_pp(X, k, 100)
    score = silhouette_coefficient(X, cluster_assgn)
    print("For k =",k,", silhouette score = ", score)
    silhouette_scores.append(score)

# Plot the results

plt.plot(range(1, 10), silhouette_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette coefficient')
plt.title('Silhouette coefficient vs. k k-means++')
plt.show()

# Question 6 - Compute the Silhouette coefficient for different values of k for Bisecting k-means
silhouette_scores = []
print("Bisecting k-means")
for k in range(1, 10):
    clustering = bisecting_kmeans(X, k)
    labels = np.concatenate([np.full(len(c), i) for i, c in enumerate(clustering)])
    score = silhouette_coefficient(X, labels)
    print("For k =",k,", silhouette score = ", score)
    silhouette_scores.append(score)

# Plot the results

plt.plot(range(1, 10), silhouette_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette coefficient')
plt.title('Silhouette coefficient vs. k Bisecting k-means')
plt.show()



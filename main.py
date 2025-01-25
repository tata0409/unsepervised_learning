import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

plt.scatter(X[:,0], X[:,1], s=50, color='gray')
plt.title('Output data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=50)
plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='x', s=200, label='centroid')
plt.title('Result of clusterization')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

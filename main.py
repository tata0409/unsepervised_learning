import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(42)
X = np.vstack((np.random.randn(50, 2) + [2, 2], np.random.randn(50, 2) + [7, 7]))

plt.scatter(X[:, 0], X[:, 1], s=50, color='gray')
plt.title("Вихідні дані")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

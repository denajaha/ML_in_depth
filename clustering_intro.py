import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# plt.scatter(X[:, 0], X[:, 1], s=150, linewidths=150)
# plt.show()

clf = KMeans(n_clusters=6)  # default is 8 dunno why

clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_  # array of the labels and they will have same index as my features ---> lowercase y from previous code
# labels are 0 or 1 because we have only 2 clusters in this case

colors = 10*['g.', 'r.','c.', 'b.', 'k.']

for i in range(len(X)):     # i is index value in this case
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=25)
plt.scatter(centroids[:, 0], centroids[:, 1],marker='x', s=150, linewidths=5)
plt.show()


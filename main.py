from kmeans import KMeans 

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

labels = kmeans.predict(X)

print("Cluster Assignments:", labels)
print("Final Centroids:", kmeans.centroids)
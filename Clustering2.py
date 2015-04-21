from numpy import loadtxt
from sklearn.cluster import DBSCAN
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer


fileName = "Book1.csv"

data = loadtxt(fileName, delimiter=';', skiprows=1)
print data

pipeline = make_pipeline(Imputer(), DBSCAN(eps=3.161, algorithm='ball_tree', min_samples=5))
pipeline.fit(data)

name, dbscan = pipeline.steps[1]
print dbscan.core_sample_indices_

print dbscan.labels_
n_clusters_ = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
print('Estimated number of clusters: %d' % n_clusters_)
n_error = 0
for i in dbscan.labels_:
    if i == -1:
        n_error += 1

print('Number of errors: %d' % n_error)

for cluster in range(n_clusters_):
    n_size = 0
    for i in dbscan.labels_:
        if i == cluster:
            n_size += 1
    print('Cluster %d size: %d' % (cluster, n_size))

print dbscan.components_
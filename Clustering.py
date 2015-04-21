from numpy import loadtxt
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer


fileName = "Book1.csv"

data = loadtxt(fileName, delimiter=';', skiprows=1)
print data

pipeline = make_pipeline(Imputer(), KMeans(n_clusters=4))
pipeline.fit(data)

name, k_means = pipeline.steps[1]
print k_means.cluster_centers_
print k_means.inertia_
print k_means.labels_



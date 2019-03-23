import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn import datasets

def loadData(filename):
	# load adult data and drop missing information
	data = pd.read_csv(filename, names=['age','workclass','fnlwgt','education',
		'education_num','marital_status','occupation','relationship','race','sex',
		'capital_gain','capital_loss','hours_per_week','native_country','result'],
		engine="python", skipinitialspace=True, na_values=['?'])
	data = data.dropna()

	# encode the labels to numbers
	eData = data.copy()
	for column in eData.columns:
		if eData.dtypes[column] == 'object':
			le = LabelEncoder()
			le.fit(eData[column])
			eData[column] = le.transform(eData[column])
	return eData

print("Load data...")
adult_data = loadData("data/adult.data")
adult_x = adult_data[['age','workclass','fnlwgt','education',
		'education_num','marital_status','occupation','relationship','race','sex',
		'capital_gain','capital_loss','hours_per_week','native_country']]
adult_y = adult_data['result']
wine = datasets.load_wine()
wine_x = wine.data

K = range(1, 40)
print("Generate model...")
# k mean model
KM_a = [KMeans(n_clusters=k).fit(adult_x) for k in K]
KM_w = [KMeans(n_clusters=k).fit(wine_x) for k in K]

print("Calculate SSE...")
# center
centroids_a = [km.cluster_centers_ for km in KM_a]
centroids_w = [km.cluster_centers_ for km in KM_w]

# calculate sum of square error
# euclidean distance
Dk_a = [cdist(adult_x, center, 'euclidean') for center in centroids_a]
cIdx_a = [np.argmin(D, axis=1) for D in Dk_a]
dist_a = [np.min(D, axis=1) for D in Dk_a]
avgWithinSS_a = [sum(d) / adult_x.shape[0] for d in dist_a]
# Total with-in sum of square
wcss_a = [sum(d**2) for d in dist_a]
tss_a = sum(pdist(adult_x)**2) / adult_x.shape[0]
bss_a = tss_a - wcss_a

Dk_w = [cdist(wine_x, center, 'euclidean') for center in centroids_w]
cIdx_w = [np.argmin(D, axis=1) for D in Dk_w]
dist_w = [np.min(D, axis=1) for D in Dk_w]
avgWithinSS_w = [sum(d) / wine_x.shape[0] for d in dist_w]
# Total with-in sum of square
wcss_w = [sum(d**2) for d in dist_w]
tss_w = sum(pdist(wine_x)**2) / wine_x.shape[0]
bss_w = tss_w - wcss_w

plt.style.use('ggplot')
# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS_a, '*-', label='Adult')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.legend(loc='best')
plt.title('KMeans clustering for Adult')
fig.savefig('figures/cluster_kmeans_adult.png')

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS_w, '*-', label='Wine')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.legend(loc='best')
plt.title('KMeans clustering for Wine')
fig.savefig('figures/cluster_kmeans_wine.png')

# use dimensionality reduction data to reproduce clustering experiments
# k = 8, change hyperparameters for dr algorithms 
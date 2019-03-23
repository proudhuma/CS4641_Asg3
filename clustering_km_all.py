import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.feature_selection import VarianceThreshold
from sklearn.random_projection import GaussianRandomProjection

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

def training(adult_data, wine_data):
	K = range(1, 20)
	print("Generate model...")
	# k mean model
	KM_a = [KMeans(n_clusters=k).fit(adult_data) for k in K]
	KM_w = [KMeans(n_clusters=k).fit(wine_data) for k in K]

	print("Calculate SSE...")
	# center
	centroids_a = [km.cluster_centers_ for km in KM_a]
	centroids_w = [km.cluster_centers_ for km in KM_w]

	# calculate sum of square error
	# euclidean distance
	Dk_a = [cdist(adult_data, center, 'euclidean') for center in centroids_a]
	cIdx_a = [np.argmin(D, axis=1) for D in Dk_a]
	dist_a = [np.min(D, axis=1) for D in Dk_a]
	avgWithinSS_a = [sum(d) / adult_data.shape[0] for d in dist_a]
	# Total with-in sum of square
	wcss_a = [sum(d**2) for d in dist_a]
	tss_a = sum(pdist(adult_data)**2) / adult_data.shape[0]
	bss_a = tss_a - wcss_a

	Dk_w = [cdist(wine_data, center, 'euclidean') for center in centroids_w]
	cIdx_w = [np.argmin(D, axis=1) for D in Dk_w]
	dist_w = [np.min(D, axis=1) for D in Dk_w]
	avgWithinSS_w = [sum(d) / wine_data.shape[0] for d in dist_w]
	# Total with-in sum of square
	wcss_w = [sum(d**2) for d in dist_w]
	tss_w = sum(pdist(wine_data)**2) / wine_data.shape[0]
	bss_w = tss_w - wcss_w
	print("done!")
	return avgWithinSS_a, avgWithinSS_w


print("Load data...")
adult_data = loadData("data/adult.data")
adult_x = adult_data[['age','workclass','fnlwgt','education',
		'education_num','marital_status','occupation','relationship','race','sex',
		'capital_gain','capital_loss','hours_per_week','native_country']]
adult_y = adult_data['result']
wine = datasets.load_wine()
wine_x = wine.data

K = range(1, 20)

# use dimensionality reduction data to reproduce clustering experiments
pca = PCA()
ica = FastICA()
wine_rp = GaussianRandomProjection(n_components=8)
adult_rp = GaussianRandomProjection(n_components=10)
fs = VarianceThreshold(threshold=0.1)

adult_x_pca = pca.fit_transform(adult_x)
adult_x_ica = ica.fit_transform(adult_x)
adult_x_rp = adult_rp.fit_transform(adult_x)
adult_x_fs = fs.fit_transform(adult_x)

wine_x_pca = pca.fit_transform(wine_x)
wine_x_ica = ica.fit_transform(wine_x)
wine_x_rp = wine_rp.fit_transform(wine_x)
wine_x_fs = fs.fit_transform(wine_x)

avgWithinSS_a, avgWithinSS_w = training(adult_x, wine_x)
avgWithinSS_a_pca, avgWithinSS_w_pca = training(adult_x_pca, wine_x_pca)
avgWithinSS_a_ica, avgWithinSS_w_ica = training(adult_x_ica, wine_x_ica)
avgWithinSS_a_rp, avgWithinSS_w_rp = training(adult_x_rp, wine_x_rp)
avgWithinSS_a_fs, avgWithinSS_w_fs = training(adult_x_fs, wine_x_fs)


plt.style.use('ggplot')
# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS_a, '*-', label='Adult')
ax.plot(K, avgWithinSS_a_pca, '*-', label='PCA')
ax.plot(K, avgWithinSS_a_ica, '*-', label='ICA')
ax.plot(K, avgWithinSS_a_fs, '*-', label='Feature Selection')
ax.plot(K, avgWithinSS_a_rp, '*-', label='Random Projection')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.legend(loc='best')
plt.title('KMeans clustering for Adult')
fig.savefig('figures/cluster_kmeans_adult.png')

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS_w, '*-', label='Adult')
ax.plot(K, avgWithinSS_w_pca, '*-', label='PCA')
ax.plot(K, avgWithinSS_w_ica, '*-', label='ICA')
ax.plot(K, avgWithinSS_w_fs, '*-', label='Feature Selection')
ax.plot(K, avgWithinSS_w_rp, '*-', label='Random Projection')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.legend(loc='best')
plt.title('KMeans clustering for Wine')
fig.savefig('figures/cluster_kmeans_wine.png')



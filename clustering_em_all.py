import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.mixture import GaussianMixture
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
	GMM_a = [GaussianMixture(n_components=k).fit(adult_data) for k in K]
	GMM_w = [GaussianMixture(n_components=k).fit(wine_data) for k in K]

	# calculate log likelihood
	LL_a = [gmm.score(adult_data) for gmm in GMM_a]
	LL_w = [gmm.score(wine_data) for gmm in GMM_w]
	# calculate BIC
	BIC_a = [gmm.bic(adult_data) for gmm in GMM_a]
	BIC_w = [gmm.bic(wine_data) for gmm in GMM_w]
	# calculate AIC
	AIC_a = [gmm.aic(adult_data) for gmm in GMM_a]
	AIC_w = [gmm.aic(wine_data) for gmm in GMM_w]
	print("done!")
	return LL_a,BIC_a,AIC_a,LL_w,BIC_w,AIC_w

print("Load data...")
adult_data = loadData("data/adult.data")
adult_x = adult_data[['age','workclass','fnlwgt','education',
		'education_num','marital_status','occupation','relationship','race','sex',
		'capital_gain','capital_loss','hours_per_week','native_country']]
adult_y = adult_data['result']
wine = datasets.load_wine()
wine_x = wine.data

K = range(1, 20)

pca = PCA()
ica = FastICA()
wine_rp = GaussianRandomProjection(n_components=8)
adult_rp = GaussianRandomProjection(n_components=10)
fs = VarianceThreshold(threshold=0.1)

adult_x_pca = fs.fit_transform(adult_x)
adult_x_ica = ica.fit_transform(adult_x)
adult_x_rp = adult_rp.fit_transform(adult_x)
adult_x_fs = fs.fit_transform(adult_x)

wine_x_pca = fs.fit_transform(wine_x)
wine_x_ica = ica.fit_transform(wine_x)
wine_x_rp = wine_rp.fit_transform(wine_x)
wine_x_fs = fs.fit_transform(wine_x)

ll_adult,bic_adult,aic_adult,ll_wine,bic_wine,aic_wine = training(adult_x, wine_x)
ll_adult_pca,bic_adult_pca,aic_adult_pca,ll_wine_pca,bic_wine_pca,aic_wine_pca = training(adult_x_pca, wine_x_pca)
ll_adult_ica,bic_adult_ica,aic_adult_ica,ll_wine_ica,bic_wine_ica,aic_wine_ica = training(adult_x_ica, wine_x_ica)
ll_adult_rp,bic_adult_rp,aic_adult_rp,ll_wine_rp,bic_wine_rp,aic_wine_rp = training(adult_x_rp, wine_x_rp)
ll_adult_fs,bic_adult_fs,aic_adult_fs,ll_wine_fs,bic_wine_fs,aic_wine_fs = training(adult_x_fs, wine_x_fs)

plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, bic_adult, '*-', label='Adult')
ax.plot(K, bic_adult_pca, '*-', label='PCA')
ax.plot(K, bic_adult_ica, '*-', label='ICA')
ax.plot(K, bic_adult_rp, '*-', label='Random Projection')
ax.plot(K, bic_adult_fs, '*-', label='Feature Selection')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Inference scores')
plt.legend(loc='best')
plt.title('BIC for Adult')
fig.savefig('figures/cluster_em_bic_adult.png')

plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, aic_adult, '*-', label='Adult')
ax.plot(K, aic_adult_pca, '*-', label='PCA')
ax.plot(K, aic_adult_ica, '*-', label='ICA')
ax.plot(K, aic_adult_rp, '*-', label='Random Projection')
ax.plot(K, aic_adult_fs, '*-', label='Feature Selection')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Inference scores')
plt.legend(loc='best')
plt.title('AIC for Adult')
fig.savefig('figures/cluster_em_aic_adult.png')

plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, ll_adult, '*-', label='Adult')
ax.plot(K, ll_adult_pca, '*-', label='PCA')
ax.plot(K, ll_adult_ica, '*-', label='ICA')
ax.plot(K, ll_adult_rp, '*-', label='Random Projection')
ax.plot(K, ll_adult_fs, '*-', label='Feature Selection')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Inference scores')
plt.legend(loc='best')
plt.title('Log Likelihood for Adult')
fig.savefig('figures/cluster_em_ll_adult.png')

plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, bic_wine, '*-', label='wine')
ax.plot(K, bic_wine_pca, '*-', label='PCA')
ax.plot(K, bic_wine_ica, '*-', label='ICA')
ax.plot(K, bic_wine_rp, '*-', label='Random Projection')
ax.plot(K, bic_wine_fs, '*-', label='Feature Selection')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Inference scores')
plt.legend(loc='best')
plt.title('BIC for Wine')
fig.savefig('figures/cluster_em_bic_wine.png')

plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, aic_wine, '*-', label='wine')
ax.plot(K, aic_wine_pca, '*-', label='PCA')
ax.plot(K, aic_wine_ica, '*-', label='ICA')
ax.plot(K, aic_wine_rp, '*-', label='Random Projection')
ax.plot(K, aic_wine_fs, '*-', label='Feature Selection')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Inference scores')
plt.legend(loc='best')
plt.title('AIC for Wine')
fig.savefig('figures/cluster_em_aic_wine.png')

plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, ll_wine, '*-', label='wine')
ax.plot(K, ll_wine_pca, '*-', label='PCA')
ax.plot(K, ll_wine_ica, '*-', label='ICA')
ax.plot(K, ll_wine_rp, '*-', label='Random Projection')
ax.plot(K, ll_wine_fs, '*-', label='Feature Selection')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Inference scores')
plt.legend(loc='best')
plt.title('Log Likelihood for Wine')
fig.savefig('figures/cluster_em_ll_wine.png')
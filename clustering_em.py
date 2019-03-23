import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.mixture import GaussianMixture

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
GMM_a = [GaussianMixture(n_components=k).fit(adult_x) for k in K]
GMM_w = [GaussianMixture(n_components=k).fit(wine_x) for k in K]

# calculate log likelihood
LL_a = [gmm.score(adult_x) for gmm in GMM_a]
LL_w = [gmm.score(wine_x) for gmm in GMM_w]
# calculate BIC
BIC_a = [gmm.bic(adult_x) for gmm in GMM_a]
BIC_w = [gmm.bic(wine_x) for gmm in GMM_w]
# calculate AIC
AIC_a = [gmm.aic(adult_x) for gmm in GMM_a]
AIC_w = [gmm.aic(wine_x) for gmm in GMM_w]

plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, BIC_a, '*-', label='BIC for Adult')
ax.plot(K, AIC_a, '*-', label='AIC for Adult')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Inference scores')
plt.legend(loc='best')
plt.title('BIC and AIC for Adult')
fig.savefig('figures/cluster_em_bic_aic_adult.png')

plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, BIC_w, '*-', label='BIC for Wine')
ax.plot(K, AIC_w, '*-', label='AIC for Wine')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Inference scores')
plt.legend(loc='best')
plt.title('BIC and AIC for Wine')
fig.savefig('figures/cluster_em_bic_aic_wine.png')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, LL_a, '*-', label='Adult')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood Curve for Adult')
plt.legend(loc='best')
fig.savefig('figures/cluster_em_log_likelihood_adult.png')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, LL_w, '*-', label='Wine')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood Curve for Wine')
plt.legend(loc='best')
fig.savefig('figures/cluster_em_log_likelihood_wine.png')


import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.feature_selection import VarianceThreshold
from sklearn.random_projection import GaussianRandomProjection

def loadData(filename):
	# load wine data and drop missing information
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
wine = datasets.load_wine()
wine_x = wine.data
wine_y = wine.target

# use dimensionality reduction data to reproduce clustering experiments
N = range(2,9)
threshold = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3]

t1 = time.time()
nncls = MLPClassifier(hidden_layer_sizes=(30,13),alpha=0.00005,learning_rate_init=0.0001,early_stopping=True,validation_fraction=0.2)
nncls.fit(wine_x, wine_y)
accuracy = nncls.score(wine_x,wine_y)
t2 = time.time()
print("NN:", accuracy, "Time:", t2-t1)


for n in N:
	em = GaussianMixture(n_components=n)
	em.fit(wine_x)
	label = em.predict(wine.data)
	reshape_l = label.reshape(label.shape[0],-1)
	new_x = np.append(wine.data, reshape_l, axis=1)

	t1 = time.time()
	nncls = MLPClassifier(hidden_layer_sizes=(30,13),alpha=0.00005,learning_rate_init=0.0001,early_stopping=True,validation_fraction=0.2)
	nncls.fit(new_x, wine_y)
	accuracy = nncls.score(new_x,wine_y)
	t2 = time.time()
	print("em:", accuracy, "Time:", t2-t1, "N:", n)
	
for n in N:
	km = KMeans(n_clusters=n)
	km.fit(wine_x)
	label = km.predict(wine.data)
	reshape_l = label.reshape(label.shape[0],-1)
	new_x = np.append(wine.data, reshape_l, axis=1)

	t1 = time.time()
	nncls = MLPClassifier(hidden_layer_sizes=(30,13),alpha=0.00005,learning_rate_init=0.0001,early_stopping=True,validation_fraction=0.2)
	nncls.fit(new_x, wine_y)
	accuracy = nncls.score(new_x,wine_y)
	t2 = time.time()
	print("k means:", accuracy, "Time:", t2-t1, "N:", n)


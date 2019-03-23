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
adult_test = loadData("data/adult.test")
adult_x = adult_data[['age','workclass','fnlwgt','education',
		'education_num','marital_status','occupation','relationship','race','sex',
		'capital_gain','capital_loss','hours_per_week','native_country']]
adult_y = adult_data['result']
adult_x_test = adult_test[['age','workclass','fnlwgt','education',
		'education_num','marital_status','occupation','relationship','race','sex',
		'capital_gain','capital_loss','hours_per_week','native_country']]
adult_y_test = adult_test['result']
wine = datasets.load_wine()
wine_x = wine.data

# use dimensionality reduction data to reproduce clustering experiments
N = range(2,9)
threshold = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3]

t1 = time.time()
nncls = MLPClassifier(hidden_layer_sizes=(30,13),alpha=0.00005,learning_rate_init=0.0001,early_stopping=True,validation_fraction=0.2)
nncls.fit(adult_x, adult_y)
accuracy = nncls.score(adult_x_test,adult_y_test)
t2 = time.time()
print("NN:", accuracy, "Time:", t2-t1)

for n in N:
	pca = PCA(n_components = n)
	
	pca_adult_x = pca.fit_transform(adult_x)
	pca_test_x = pca.fit_transform(adult_x_test)

	t1 = time.time()
	nncls = MLPClassifier(hidden_layer_sizes=(30,13),alpha=0.00005,learning_rate_init=0.0001,early_stopping=True,validation_fraction=0.2)
	nncls.fit(pca_adult_x, adult_y)
	accuracy = nncls.score(pca_test_x,adult_y_test)
	t2 = time.time()
	print("PCA:", accuracy, "Time:", t2-t1, "N:", n)

for n in N:	
	ica = FastICA(n_components=n)

	ica_adult_x = ica.fit_transform(adult_x)
	ica_test_x = ica.fit_transform(adult_x_test)

	t1 = time.time()
	nncls = MLPClassifier(hidden_layer_sizes=(30,13),alpha=0.00005,learning_rate_init=0.0001,early_stopping=True,validation_fraction=0.2)
	nncls.fit(ica_adult_x, adult_y)
	accuracy = nncls.score(ica_test_x,adult_y_test)
	t2 = time.time()
	print("ICA:", accuracy, "Time:", t2-t1, "N:", n)
	
for n in N:
	adult_rp = GaussianRandomProjection(n_components=10)

	rp_adult_x = adult_rp.fit_transform(adult_x)
	rp_test_x = adult_rp.fit_transform(adult_x_test)

	t1 = time.time()
	nncls = MLPClassifier(hidden_layer_sizes=(30,13),alpha=0.00005,learning_rate_init=0.0001,early_stopping=True,validation_fraction=0.2)
	nncls.fit(rp_adult_x, adult_y)
	accuracy = nncls.score(rp_test_x,adult_y_test)
	t2 = time.time()
	print("Random Projection:", accuracy, "Time:", t2-t1, "N:", n)

for theta in threshold:
	fs = VarianceThreshold(threshold=theta)

	fs_adult_x = fs.fit_transform(adult_x)
	fs_test_x = fs.fit_transform(adult_x_test)	

	t1 = time.time()
	nncls = MLPClassifier(hidden_layer_sizes=(30,13),alpha=0.00005,learning_rate_init=0.0001,early_stopping=True,validation_fraction=0.2)
	nncls.fit(fs_adult_x, adult_y)
	accuracy = nncls.score(fs_test_x,adult_y_test)
	t2 = time.time()
	print("Feature Selection:", accuracy, "Time:", t2-t1, "Theta:", theta)


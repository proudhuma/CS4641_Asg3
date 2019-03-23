import csv
import numpy as np
import pandas as pd
from collections import defaultdict
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.feature_selection import VarianceThreshold
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import datasets

import seaborn as sns

C = ['#ff400e',
'#f5892f',
'#20a8f6',
'#ebd250',
'#17f8b8',
'#1cc8dd',
'#6b0693',
'#307eea',
'#8dea7d',
'#5436b6',
'#8923a2']

sns.set(color_codes=True)

df = datasets.load_wine()

# Train set
df_y = df.target
df_x = df.data

##########################################################################

nrange = [2,5,7,9,11]


print ("PCA - kmeans")
for n in nrange:
    pca = PCA(n_components=n)
    pca.fit(df_x)

    # print "Score:", pca.score(df)
    print ("N:", n)
    print ("Variance:", pca.noise_variance_)

    reduced_data = PCA(n_components=n).fit_transform(df_x)
    kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
    kmeans.fit(reduced_data)

    correct = 0
    for i in range(10):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[label] == float(i):
                lab = kmeans.predict([reduced_data[index]])
                d[lab[0]] += 1
        if d: correct += max(d.values())
    
    print ("Accuracy:", correct)

print ("PCA - EM")
for n in nrange:
    pca = PCA(n_components=n)
    pca.fit(df_x)

    # print "Score:", pca.score(df)
    print ("N:", n)
    print ("Variance:", pca.noise_variance_)

    reduced_data = PCA(n_components=n).fit_transform(df_x)
    em = GaussianMixture(n_components=10)
    em.fit(reduced_data)

    correct = 0
    for i in range(10):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[label] == float(i):
                lab = em.predict([reduced_data[index]])
                d[lab[0]] += 1
        if d: correct += max(d.values())
    
    print ("Accuracy:", correct)

# generate graph
reduced_data = PCA(n_components=2).fit_transform(df_x)
kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
kmeans.fit(reduced_data)

# Graph
h = .02

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

fig = plt.figure(1)
plt.clf()

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

for index, row in df.iterrows():
    col = kmeans.predict([reduced_data[index,:]])
    plt.plot(reduced_data[index, 0], reduced_data[index, 1], '.', color=C[2*col[0]], markersize=2)

plt.title('K-means clustering on the Wine dataset (PCA-reduced data)')

plt.xticks(())
plt.yticks(())
fig.savefig('figures/wine_km_PCA.png')
plt.close(fig)

#########################################################################

# fig = plt.figure(2)
# plt.clf()

# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)

# for index, row in df.iterrows():
#     plt.plot(reduced_data[index, 0], reduced_data[index, 1], '.', color=C[2*int(row[label])], markersize=2)

# plt.xticks(())
# plt.yticks(())

# plt.title('Each label mapped on the PCA-reduced 2D graph (Wine)')
# fig.savefig('figures/wine_km_PCA_rankings.png')
# plt.close(fig)

############################################################################

# reduced_data = PCA(n_components=2).fit_transform(df_x)
# em = GaussianMixture(n_components=10)
# em.fit(reduced_data)

# # # Graph
# h = .02

# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# fig = plt.figure(1)
# plt.clf()

# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)

# for index, row in df.iterrows():
#     col = em.predict([reduced_data[index,:]])
#     plt.plot(reduced_data[index, 0], reduced_data[index, 1], '.', color=C[col[0]], markersize=2)

# plt.title('EM clustering on the Wine dataset (PCA-reduced data)')

# plt.xticks(())
# plt.yticks(())
# fig.savefig('figures/wine_em_PCA.png')
# plt.close(fig)

# ##########################################################################

# fig = plt.figure(2)
# plt.clf()

# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)

# for index, row in df.iterrows():
#     plt.plot(reduced_data[index, 0], reduced_data[index, 1], '.', color=C[int(row[label])], markersize=2)

# plt.xticks(())
# plt.yticks(())

# plt.title('Each label mapped on the PCA-reduced 2D graph (Wine)')
# fig.savefig('figures/wine_em_PCA_rankings.png')
# plt.close(fig)

##########################################################################

# print "ICA - kmeans"
# for n in nrange:

#     # print "Score:", pca.score(df)
#     print "N:", n

#     reduced_data = FastICA(n_components=n).fit_transform(df_x)
#     kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
#     kmeans.fit(reduced_data)

#     correct = 0
#     for i in range(10):
#         d = defaultdict(int)
#         for index, row in df.iterrows():
#             if row[label] == float(i):
#                 lab = kmeans.predict([reduced_data[index]])
#                 d[lab[0]] += 1
#         if d: correct += max(d.values())
    
#     print "Accuracy:", float(correct) / 4898

# print "ICA - EM"
# for n in nrange:

#     # print "Score:", pca.score(df)
#     print "N:", n

#     reduced_data = FastICA(n_components=n).fit_transform(df_x)
#     em = GaussianMixture(n_components=10)
#     em.fit(reduced_data)

#     correct = 0
#     for i in range(10):
#         d = defaultdict(int)
#         for index, row in df.iterrows():
#             if row[label] == float(i):
#                 lab = em.predict([reduced_data[index]])
#                 d[lab[0]] += 1
#         if d: correct += max(d.values())
    
#     print "Accuracy:", float(correct) / 4898

# reduced_data = FastICA(n_components=2).fit_transform(df_x)
# em = GaussianMixture(n_components=10)
# em.fit(reduced_data)

# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 0.01, reduced_data[:, 0].max() + 0.01
# y_min, y_max = reduced_data[:, 1].min() - 0.01, reduced_data[:, 1].max() + 0.01

# fig = plt.figure(3)
# plt.clf()

# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)

# for index, row in df.iterrows():
#     col = em.predict([reduced_data[index,:]])
#     plt.plot(reduced_data[index, 0], reduced_data[index, 1], '.', color=C[col[0]], markersize=3)

# plt.title('Expectation Maximization clustering on the Wine dataset (ICA-reduced data)')

# plt.xticks(())
# plt.yticks(())
# fig.savefig('figures/wine_em_ICA.png')
# plt.close(fig)

# # ##########################################################################

# fig = plt.figure(4)
# plt.clf()

# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)

# for index, row in df.iterrows():
#     plt.plot(reduced_data[index, 0], reduced_data[index, 1], '.', color=C[int(row[label])], markersize=3)

# plt.xticks(())
# plt.yticks(())

# plt.title('Labels mapped on the ICA-reduced 2D graph')
# fig.savefig('figures/wine_em_ICA_rankings.png')

# plt.close(fig)

# # ##########################################################################

# print "RP - kmeans"
# for n in nrange:
#     transformer = GaussianRandomProjection(n_components=n)
#     reduced_data = transformer.fit_transform(df_x)

#     print "N:", n

#     kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
#     kmeans.fit(reduced_data)

#     correct = 0
#     for i in range(10):
#         d = defaultdict(int)
#         for index, row in df.iterrows():
#             if row[label] == float(i):
#                 lab = kmeans.predict([reduced_data[index]])
#                 d[lab[0]] += 1
#         if d: correct += max(d.values())
    
#     print "Accuracy:", float(correct) / 4898

# print "RP - EM"
# for n in nrange:
#     transformer = GaussianRandomProjection(n_components=n)
#     reduced_data = transformer.fit_transform(df_x)

#     print "N:", n

#     kmeans = GaussianMixture(n_components=10)
#     kmeans.fit(reduced_data)

#     correct = 0
#     for i in range(10):
#         d = defaultdict(int)
#         for index, row in df.iterrows():
#             if row[label] == float(i):
#                 lab = kmeans.predict([reduced_data[index]])
#                 d[lab[0]] += 1
#         if d: correct += max(d.values())
    
#     print "Accuracy:", float(correct) / 4898

# # ##########################################################################
# df_x.dropna()
# varss = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# print "Feature Selection - kmeans"
# for var in varss:
#     sel = VarianceThreshold(threshold=var)
#     reduced_data = sel.fit_transform(df_x)

#     print "Var:", var

#     kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
#     kmeans.fit(reduced_data)

#     correct = 0
#     for i in range(10):
#         d = defaultdict(int)
#         for index, row in df.iterrows():
#             if row[label] == float(i):
#                 lab = kmeans.predict([reduced_data[index]])
#                 d[lab[0]] += 1
#         if d: correct += max(d.values())
    
#     print "Accuracy:", float(correct) / 4898

# print "Feature Selection - EM"
# for var in varss:
#     sel = VarianceThreshold(threshold=var)
#     reduced_data = sel.fit_transform(df_x)

#     print "Var:", var

#     kmeans = GaussianMixture(n_components=10)
#     kmeans.fit(reduced_data)

#     correct = 0
#     for i in range(10):
#         d = defaultdict(int)
#         for index, row in df.iterrows():
#             if row[label] == float(i):
#                 lab = kmeans.predict([reduced_data[index]])
#                 d[lab[0]] += 1
#         if d: correct += max(d.values())
    
#     print "Accuracy:", float(correct) / 4898

# ##########################################################################
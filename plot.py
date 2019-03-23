import numpy as np
import matplotlib.pyplot as plt

N = range(2,9)
NN = [0.7818 for i in range(2,9)]
NN_time = [1.2437 for i in range(2,9)]
pca = [0.7796,0.7815,0.7674,0.7839,0.7832,0.7699,0.7863]
pca_time = [0.8417, 2.7536, 1.8281, 1.6156, 2.9660, 2.9730, 5.6778]
ica = [0.7543, 0.7543 ,0.7543 ,0.7543 ,0.7543 ,0.7543 ,0.7543 ]
ica_time = [0.8626,0.9384,1.0272,0.8208,0.9345,1.1948,0.9454]
rp = [0.7543,0.2456,0.2443,0.2355,0.7543,0.2458,0.2456]
rp_time =[1.1050,0.8397,1.4211,2.1871,2.5741,1.2067,1.0511]
threshold = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3]
fs = [0.7925,0.7907,0.7836,0.7861,0.7871,0.7644,0.7816,0.7842,0.7941,0.7871,0.7842,0.7763]
fs_time = [2.018,1.909,1.487,1.306,0.855,1.437,1.010,1.678,1.378,2.572,2.563,0.907]

plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(N, NN, '*-', label='NN')
ax.plot(N, pca, '*-', label='PCA')
ax.plot(N, ica, '*-', label='ICA')
ax.plot(N, rp, '*-', label='Random Projection')
plt.grid(True)
plt.xlabel('N')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title('Neural Network after Dimension Reduction')
fig.savefig('figures/adult_nn_dr.png')

plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(N, NN_time, '*-', label='NN')
ax.plot(N, pca_time, '*-', label='PCA')
ax.plot(N, ica_time, '*-', label='ICA')
ax.plot(N, rp_time, '*-', label='Random Projection')
plt.grid(True)
plt.xlabel('N')
plt.ylabel('Time')
plt.legend(loc='best')
plt.title('Time for Neural Network after Dimension Reduction')
fig.savefig('figures/adult_nn_dr_time.png')

N = range(2,9)
NN = [0.2697 for i in range(2,9)]
NN_time = [0.01394 for i in range(2,9)]
pca = [0.0674,0.6797,0.3314,0.3651,0.4213,0.6741,0.3258]
pca_time = [0.01097,0.00897,0.00997,0.01097,0.01396,0.01395,0.00897]
ica = [0.3988,0.2696,0.3370,0.3988,0.2696,0.3314,0.2696]
ica_time = [0.00997,0.01097,0.00797,0.00897,0.01096,0.00897,0.01097]
rp = [0.3988,0.3988,0.3595,0.0674,0.2696,0.2696,0.3314]
rp_time = [0.01296,0.01196,0.00997,0.01296,0.01096,0.01994,0.01296]
threshold = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3]
fs = [0.3988,0.2808,0.3988,0.3988,0.4943,0.2696,0.3314,0.2696,0.3988,0.3988,0.3988,0.3314]
fs_time = [0.01097,0.01795,0.01097,0.00897,0.01795,0.01396,0.01097,0.01097,0.00997,0.01296,0.01197,0.01096]

plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(N, NN, '*-', label='NN')
ax.plot(N, pca, '*-', label='PCA')
ax.plot(N, ica, '*-', label='ICA')
ax.plot(N, rp, '*-', label='Random Projection')
plt.grid(True)
plt.xlabel('N')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title('Neural Network after Dimension Reduction')
fig.savefig('figures/wine_nn_dr.png')

plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(N, NN_time, '*-', label='NN')
ax.plot(N, pca_time, '*-', label='PCA')
ax.plot(N, ica_time, '*-', label='ICA')
ax.plot(N, rp_time, '*-', label='Random Projection')
plt.grid(True)
plt.xlabel('N')
plt.ylabel('Time')
plt.legend(loc='best')
plt.title('Time for Neural Network after Dimension Reduction')
fig.savefig('figures/wine_nn_dr_time.png')

N = range(2,9)
NN = [0.3314 for i in range(2,9)]
NN_time = [0.02294 for i in range(2,9)]
em = [0.26966,0.3314,0.3988,0.2696,0.3988,0.3988,0.1629]
em_time = [0.01495,0.01196,0.00897,0.00997,0.00997,0.01097,0.01396]
km = [0.3988,0.2696,0.3988,0.3314,0.3988,0.2696,0.3988]
km_time = [0.01196,0.01495,0.00997,0.00997,0.01196,0.00797,0.01097]

plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(N, NN, '*-', label='NN')
ax.plot(N, em, '*-', label='Expectation Maximization')
ax.plot(N, km, '*-', label='K Means')
plt.grid(True)
plt.xlabel('N')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title('Neural Network after Clustering')
fig.savefig('figures/wine_nn_cluster.png')

plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(N, NN_time, '*-', label='NN')
ax.plot(N, em_time, '*-', label='Expectation Maximization')
ax.plot(N, km_time, '*-', label='K Means')
plt.grid(True)
plt.xlabel('N')
plt.ylabel('Time')
plt.legend(loc='best')
plt.title('Time for Neural Network after Clustering')
fig.savefig('figures/wine_nn_cluster_time.png')
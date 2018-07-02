from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D

datas = datasets.load_breast_cancer()

X = np.asarray(datas.data)

features = 3

dat = np.array(X, dtype = np.float32)

def distance(a, b, ax = 1):
    return np.linalg.norm(a-b, axis = ax)

#initialize number of clusters
cluster = 5

#initialize the centroid set
C[i] = np.random.randint(np.min(dat[i], np.max[i]) for i in range(features))

#cx = np.random.randint(np.min(X), np.max(X), size = cluster)
#cy = np.random.randint(np.min(Y), np.max(Y), size = cluster)
#cz = np.random.randint(np.min(Y), np.max(Y), size = cluster)
#C = np.array(list(zip(cx, cy)), dtype=np.float32)

print(C)

C_old = np.zeros(C.shape)

clusters = np.zeros(len(X))

improv = distance(C, C_old, None)

while improv !=0:
    for i in range(len(dat)):
        distances = distance(X[i], C)
        c = np.argmin(distances)
        clusters[i] = c
    C_old = deepcopy(C)

    for i in range(cluster):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        if(len(points) == 0):
            C = np.random.randint(np.min(dat[i], np.max[i]), size = np.size(C), dtype = float32)

        else:
            C[i] = np.mean(points, axis=0)
    improv = distance(C, C_old, None)

#graph data and centroids
plt.rcParams['figure.figsize'] = (5, 3)
colors = ['#efd1c9', '#cdf4c1', '#bed3f4', '#f2f2d0', '#c2efe9', '#efc2e6', '#bff756','#b691bc']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for h in range(cluster):
        points = np.array([dat[f] for f in range(len(X)) if clusters[f] == h])
        ax.scatter(points[:, 0], points[:, 1], points[:,2], s=7, c=colors[h])

ax.scatter(C[:, 0], C[:, 1], C[:,2], marker='*', s=200, c='#050505')

plt.show()
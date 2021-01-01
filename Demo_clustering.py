# -*- coding: utf-8 -*-
"""
@author:
    
Saglam, Ali, & Baykan, N. A. (2017).

"Sequential image segmentation based on min- imum spanning tree representation".

Pattern Recognition Letters, 87 , 155–162.

https://doi.org/10.1016/j.patrec.2016.06.001 .
"""

from sequential_clustering import sequential_clustering
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler

features, true_labels = make_moons(n_samples=250, noise=0.05, random_state=42)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

############### S_MST CLUSTERING ##################

labels = sequential_clustering(scaled_features, m = 3, l = "scale")

###################################################

plt.scatter(features[:,0], features[:,1], cmap='rainbow', c=labels)
plt.plot()

  
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

# 通过k-means选择锚点
def sampling_kmeans(X, anchor_num=50):
    inds = []
    for x in X:
        KM = KMeans(n_clusters=anchor_num, random_state=1234).fit(x)
        anchor = KM.cluster_centers_
        inds.append(anchor)

    return inds

# 通过k-means++选择锚点
def sampling_minikmeans(X, anchor_num=50):
    inds = []
    for x in X :
        KM = MiniBatchKMeans(init="k-means++", n_clusters=anchor_num, random_state=1234, batch_size=1000).fit(x)
        anchor = KM.cluster_centers_
        inds.append(anchor)

    return inds

# 通过随机采样选择锚点
def sampling_random(X, anchor_num=50):
    inds = []
    for x in X:
        N = x.shape[0]
        ind = np.random.choice([i for i in range(N)], anchor_num)
        inds.append(x[ind])

    return inds



import scipy.io
import numpy as np
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


# # 加载 .mat 文件
# data = scipy.io.loadmat('./data/mat/ACM.mat')
#
# # 查看文件中的变量
# print(data)


# data2 = np.load('./data/npz/amazon_computers.npz')
# print(data2)

# # .mat文件转换为.npz文件
# mat_data = scipy.io.loadmat('./data/mat/AMAP.mat')
#
# # 将数据保存为npz格式
# # 假设mat_data是一个字典，包含了MAT文件中的所有变量
# np.savez('./data/npz/amazon_photos.npz', **mat_data)

# dataset = "./data/mat/" + 'ACM'
# data = scipy.io.loadmat('{}.mat'.format(dataset))
#
# X = data['features']
# A = data['PAP']
# B = data['PLP']
#
# A = A.toarray()
# # # 假设 A 是邻接矩阵
# # A = np.array(A)
# #
# # 打印数组的形状
# print("Shape of A:", A.shape)
#
# # 计算每个节点的度（按行求和）
# Dr = np.sum(A, axis=1)
#
# # 输出每个节点的度
# print("Degrees of each node:", Dr)


# 通过k-means选择锚点
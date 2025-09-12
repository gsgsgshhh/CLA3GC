import numpy as np
from sklearn.decomposition import PCA, SparsePCA

# 对邻接矩阵A进行归一化处理
def normalize_matrix(A):
    D = np.sum(A, axis=1) # 计算每一行的和，即每个节点的度（D是个列向量）
    D = np.power(D, -0.5) # 取倒数平方根
    D[np.isinf(D)] = 0 # 处理无穷大的值
    D[np.isnan(D)] = 0 # 处理NAN
    D = np.diagflat(D) # 将向量D转换为一个对角矩阵D（对角线上）
    A = D.dot(A).dot(D) # 对A归一化
    return A

# 对特征矩阵X进行归一化处理
def normalize_feature(X):
    X_new = [x / np.linalg.norm(x) for x in X] # 计算每一行的l2范数，使每个向量的模长变为 1
    X_new = np.array(X_new)
    return X_new



'''
# 这种方法更快一些
def normalize_feature(X):
    # 计算每行的 L2 范数 (axis=1 表示按行计算)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # 使用广播机制进行归一化
    X_new = X / norms
    return X_new

'''

def dimension_reduction(X, dim=64, idd=0, dt="ACM"):
    try:
        X_new = np.load("./feature_matrix/{}_X_{}.npy".format(dt, idd))
    except Exception:
        try:
            # print(8 / 0)
            pca = PCA(n_components=dim, random_state=12345)
            X_new = pca.fit_transform(X)
        except:  # !! this for extra large-scale data
            done = 0
            batch_size = 1000000
            N = X.shape[0]
            while done <= 0:
                try:
                    X_new = []
                    # batch_size = 1000000
                    steps = int(N / batch_size) + 1
                    for step in range(steps):
                        try:
                            x_new = np.load("./feature_matrix/{}_X_{}_{}.npy".format(dt, idd, step))
                        except Exception:
                            x = X[step * batch_size: min((step + 1) * batch_size, N)]
                            pca = PCA(n_components=dim, random_state=12345)
                            x_new = pca.fit_transform(x)
                        finally:
                            assert x_new != None
                            X_new.append(x_new)
                    done += 1

                except:
                    batch_size = batch_size / 2
                    pass

    return X_new



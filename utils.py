import numpy as np
from sklearn.decomposition import PCA, SparsePCA

def normalize_matrix(A):
    D = np.sum(A, axis=1)
    D = np.power(D, -0.5)
    D[np.isinf(D)] = 0
    D[np.isnan(D)] = 0
    D = np.diagflat(D)
    A = D.dot(A).dot(D)
    return A

def normalize_feature(X):
    X_new = [x / np.linalg.norm(x) for x in X]
    X_new = np.array(X_new)
    return X_new

def dimension_reduction(X, dim=64, idd=0, dt="ACM"):
    try:
        X_new = np.load("./feature_matrix/{}_X_{}.npy".format(dt, idd))
    except Exception:
        try:
            pca = PCA(n_components=dim, random_state=12345)
            X_new = pca.fit_transform(X)
        except:
            done = 0
            batch_size = 1000000
            N = X.shape[0]
            while done <= 0:
                try:
                    X_new = []
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




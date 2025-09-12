from math import gamma
import numpy as np
from scipy.linalg import solve_sylvester
import warnings

warnings.filterwarnings('ignore')

def Effecient_clustering(H, B, alpha=1, beta=1, epochs=100, threshold=1e-5):
    Im = np.eye(B.shape[0])
    loss_last = 1e16

    for epoch in range(epochs):
        BBt = B.dot(B.T)
        BHt = B.dot(H.T)
        tmp1 = BBt + (alpha + beta) * Im
        tmp2 = np.linalg.inv(tmp1).dot(BHt)
        S = tmp2 * (1 + alpha)
        loss_SE = np.linalg.norm(H.T - B.T.dot(S), 'fro') ** 2
        loss_L2 = np.linalg.norm(S, 'fro') ** 2
        loss_inv = np.linalg.norm(B.dot(H.T) - S, 'fro') ** 2
        loss_total = loss_SE + alpha * loss_L2 + beta * loss_inv
        if loss_last - loss_total < threshold * loss_last:
            break
        else:
            loss_last = loss_total
        SSt = S.dot(S.T)
        HtH = H.T.dot(H)
        SH = S.dot(H)
        B = solve_sylvester(SSt, beta * HtH, SH * (1 + alpha))
    return S, B


def New_Effecient_clustering(H, B, alpha=1, beta=1, gamma=1, epochs=100, threshold=1e-5):
    Im = np.eye(B.shape[0])
    loss_last = 1e16
    for epoch in range(epochs):
        BBt = B.dot(B.T)
        BHt = B.dot(H.T)
        B_sq = np.sum(B ** 2, axis=1).reshape(-1, 1)
        H_sq = np.sum(H ** 2, axis=1).reshape(1, -1)
        BH_dot = np.dot(B, H.T)
        Y = np.sum(B_sq) + np.sum(H_sq) - 2 * np.sum(BH_dot)
        tmp1 = BBt + (alpha + beta) * Im
        tmp2 = np.linalg.inv(tmp1)
        tmp3 = (1 + alpha) * BHt - gamma/2 * Y
        S = tmp2.dot(tmp3)
        loss_SE = np.linalg.norm(H.T - B.T.dot(S), 'fro') ** 2
        loss_L2 = np.linalg.norm(S, 'fro') ** 2
        loss_inv = np.linalg.norm(B.dot(H.T) - S, 'fro') ** 2
        loss_L4 = np.trace(np.dot(S, Y.T))
        loss_total = loss_SE + alpha * loss_L2 + beta * loss_inv + gamma * loss_L4
        if loss_last - loss_total < threshold * loss_last:
            break
        else:
            loss_last = loss_total
        SSt = S.dot(S.T)
        HtH = H.T.dot(H)
        SH = S.dot(H)
        B = solve_sylvester(SSt, beta * HtH, SH * (1 + alpha))
    return S, B


def Effecient_multi_view_clustering(H, B, alpha=1, beta=1, eps=1e-5, threshold=1e-5, epochs=100):
    V = len(H)
    Im = np.eye(B[0].shape[0])
    omiga = [1 / V] * V
    loss_last = 1e16
    for epoch in range(epochs):
        BBt = []
        BHt = []
        for v in range(V):
            BBt_v = B[v].dot(B[v].T)
            BHt_v = B[v].dot(H[v].T)
            BBt.append(BBt_v)
            BHt.append(BHt_v)
        tmp1 = 0
        tmp2 = 0
        for v in range(V):
            omiga_v_square = omiga[v] ** 2
            tmp1 += omiga_v_square * (BBt[v] + alpha * Im)
            tmp2 += omiga_v_square * BHt[v]
        S = np.linalg.inv(tmp1 + beta * Im).dot(tmp2 * (1 + alpha))
        loss_view = 0
        for v in range(V):
            loss_SE = np.linalg.norm(H[v].T - B[v].T.dot(S), 'fro') ** 2
            loss_BG = np.linalg.norm(BHt[v] - S, 'fro') ** 2
            omiga_v_square = omiga[v] ** 2
            loss_view += omiga_v_square * (loss_SE + alpha * loss_BG)
        loss_L2 = np.linalg.norm(S, 'fro') ** 2
        loss_total = loss_view + beta * loss_L2
        if loss_last - loss_total < threshold * loss_last:
            break
        else:
            loss_last = loss_total
        SSt = S.dot(S.T)
        for v in range(V):
            HtH_v = H[v].T.dot(H[v])
            SH_v = S.dot(H[v])
            B[v] = solve_sylvester(SSt, alpha * HtH_v, SH_v * (1 + alpha))
        Const_loss = np.zeros(V)
        for v in range(V):
            SE = np.linalg.norm(H[v].T - B[v].T.dot(S), 'fro') ** 2
            BG = np.linalg.norm(BHt[v] - S, 'fro') ** 2
            Const_loss[v] += SE + alpha * BG
        Total = [1 / CL for CL in Const_loss]
        Total_sum = np.array(Total).sum()
        for v in range(V):
            omiga[v] = (Total[v]) / (Total_sum + eps)
    return S, omiga, B

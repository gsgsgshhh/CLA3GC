from math import gamma
import numpy as np
from scipy.linalg import solve_sylvester
import warnings
from sympy.abc import alpha
warnings.filterwarnings('ignore')

# def Efficient_clustering(H, B, alpha=1, beta=1, epochs=100, threshold=1e-5):
#     Im = np.eye(B.shape[0])  
#     loss_last = 1e16 
#     for epoch in range(epochs):
#         BBt = B.dot(B.T)
#         BHt = B.dot(H.T)
#         B_sq = np.sum(B ** 2, axis=1, keepdims=True)  # shape: (m, 1)
#         H_sq = np.sum(H ** 2, axis=1, keepdims=True).T  # shape: (1, n)
#         BH_dot = np.dot(B, H.T)  # shape: (m, n)
#         Y = B_sq + H_sq - 2 * BH_dot  # shape: (m, n)
#         tmp1 = np.linalg.inv(BBt +  alpha * Im)
#         tmp2 = (1+alpha) * BHt - beta/2 * Y
#         S = tmp1.dot(tmp2)
#         loss_SE = np.linalg.norm(H.T - B.T.dot(S), 'fro') ** 2  
#         loss_2 = np.linalg.norm(B.dot(H.T) - S, 'fro') ** 2  
#         loss_3 = np.sum(S * Y)
#         loss_total = loss_SE + alpha * loss_2 + beta * loss_3
#         if loss_last - loss_total < threshold * loss_last:
#             break
#         else:
#             loss_last = loss_total
#         SSt = S.dot(S.T)
#         HtH = H.T.dot(H)
#         SH = S.dot(H)
#         B = solve_sylvester(SSt, alpha * HtH, SH * (1 + alpha))
#     return S, B

# def Efficient_clustering(H, B, alpha=1, beta=1, epochs=100, threshold=1e-5):
#     Im = np.eye(B.shape[0]) 
#     loss_last = 1e16  
#     for epoch in range(epochs):
#         BBt = B.dot(B.T)
#         BHt = B.dot(H.T)
#         tmp1 = BBt + (alpha + beta) * Im
#         tmp2 = np.linalg.inv(tmp1).dot(BHt)
#         S = tmp2 * (1 + alpha)
#         loss_SE = np.linalg.norm(H.T - B.T.dot(S), 'fro') ** 2  
#         loss_L2 = np.linalg.norm(S, 'fro') ** 2  
#         loss_inv = np.linalg.norm(B.dot(H.T) - S, 'fro') ** 2  
#         loss_total = loss_SE + beta * loss_L2 + alpha * loss_inv
#         if loss_last - loss_total < threshold * loss_last:
#             break
#         else:
#             loss_last = loss_total
#         SSt = S.dot(S.T)
#         HtH = H.T.dot(H)
#         SH = S.dot(H)
#         B = solve_sylvester(SSt, alpha * HtH, SH * (1 + alpha))
#
#     return S, B


# def Efficient_clustering(H, B, alpha=1, beta=1, epochs=100, threshold=1e-5):
#     Im = np.eye(B.shape[0]) 
#     loss_last = 1e16 
#     for epoch in range(epochs):
#         BBt = B.dot(B.T)
#         BHt = B.dot(H.T)
#         B_sq = np.sum(B ** 2, axis=1, keepdims=True)  # shape: (m, 1)
#         H_sq = np.sum(H ** 2, axis=1, keepdims=True).T  # shape: (1, n)
#         BH_dot = np.dot(B, H.T)  # shape: (m, n)
#         Y = B_sq + H_sq - 2 * BH_dot  # shape: (m, n)
#         tmp1 = np.linalg.inv(BBt + alpha*Im)
#         tmp2 = BHt - beta/2 * Y
#         S = tmp1.dot(tmp2)
#         loss_SE = np.linalg.norm(H.T - B.T.dot(S), 'fro') ** 2
#         loss_2 = np.linalg.norm(S, 'fro') ** 2
#         loss_3 = np.sum(S * Y)
#         loss_total = loss_SE + alpha * loss_2 + beta * loss_3
#         if loss_last - loss_total < threshold * loss_last:
#             break
#         else:
#             loss_last = loss_total
#         SSt = S.dot(S.T)
#         SH = S.dot(H)
#         B = SSt.dot(SH)
#
#     return S, B


# def Efficient_clustering(H, B, alpha=1, beta=1, epochs=100, threshold=1e-5):
#     Im = np.eye(B.shape[0])  
#     loss_last = 1e16 
#     for epoch in range(epochs):
#         BBt = B.dot(B.T)
#         BHt = B.dot(H.T)
#         tmp1 = np.linalg.inv(BBt + alpha*Im)
#         S = tmp1.dot(BHt)
#         print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#         print(S.shape)
#         loss_SE = np.linalg.norm(H.T - B.T.dot(S), 'fro') ** 2
#         loss_2 = np.linalg.norm(S, 'fro') ** 2
#         loss_total = loss_SE + alpha * loss_2
#         if loss_last - loss_total < threshold * loss_last:
#             break
#         else:
#             loss_last = loss_total
#         SSt = S.dot(S.T)
#         SH = S.dot(H)
#         B = SSt.dot(SH)
#
#     return S, B

def New_Efficient_clustering(H, B, alpha=1, beta=1, gamma=1, epochs=100, threshold=1e-5):
    Im = np.eye(B.shape[0])  
    loss_last = 1e16  
    for epoch in range(epochs):
        BBt = B.dot(B.T)
        BHt = B.dot(H.T)
        B_sq = np.sum(B ** 2, axis=1, keepdims=True)  # shape: (m, 1)
        H_sq = np.sum(H ** 2, axis=1, keepdims=True).T  # shape: (1, n)
        BH_dot = np.dot(B, H.T)  # shape: (m, n)
        Y = B_sq + H_sq - 2 * BH_dot  # shape: (m, n)
        tmp1 = BBt + (alpha + gamma) * Im
        tmp2 = np.linalg.inv(tmp1)
        tmp3 = (1 + alpha) * BHt - beta/2 * Y
        S = tmp2.dot(tmp3)
        loss_SE = np.linalg.norm(H.T - B.T.dot(S),'fro') ** 2
        loss_inv = np.linalg.norm(B.dot(H.T) - S, 'fro') ** 2
        loss_L3 = np.sum(S * Y)
        loss_L4 = np.linalg.norm(S, 'fro') ** 2
        loss_total = loss_SE + alpha * loss_inv + beta * loss_L3 + gamma * loss_L4
        if loss_last - loss_total < threshold * loss_last:
            break
        else:
            loss_last = loss_total
        SSt = S.dot(S.T)
        HtH = H.T.dot(H)
        SH = S.dot(H)
        B = solve_sylvester(SSt,  alpha * HtH, SH * (1 + alpha))

    return S, B

def Efficient_multi_view_clustering(H, B, alpha=1, beta=1, gamma=1, eps=1e-5, threshold=1e-5, epochs=100):
    V = len(H)
    Im = np.eye(B[0].shape[0])
    omiga = [1 / V] * V
    loss_last = 1e16
    lo =[]
    for epoch in range(epochs):

        BBt = []
        BHt = []
        Y = []
        for v in range(V):
            BBt_v = B[v].dot(B[v].T)
            BHt_v = B[v].dot(H[v].T)

            BBt.append(BBt_v)
            BHt.append(BHt_v)
            B_sq = np.sum(B[v] ** 2, axis=1, keepdims=True)
            H_sq = np.sum(H[v] ** 2, axis=1, keepdims=True).T
            BH_dot = np.dot(B[v], H[v].T)
            Y_v = B_sq + H_sq - 2 * BH_dot
            Y.append(Y_v)

        tmp1 = 0
        tmp2 = 0
        for v in range(V):
            omiga_v_square = omiga[v] ** 2
            tmp1 += omiga_v_square * (BBt[v] + alpha * Im)
            tmp2 += omiga_v_square * ((1+alpha)*BHt[v] - beta/2 * Y[v])

        S = np.linalg.inv(tmp1 + gamma * Im).dot(tmp2)

        loss_view = 0
        for v in range(V):
            loss_SE = np.linalg.norm(H[v].T - B[v].T.dot(S), 'fro') ** 2
            loss_BG = np.linalg.norm(BHt[v] - S, 'fro') ** 2
            loss_TG = np.sum(S * Y[v])
            omiga_v_square = omiga[v] ** 2
            loss_view += omiga_v_square * (loss_SE + alpha * loss_BG + beta * loss_TG)

        loss_L2 = np.linalg.norm(S, 'fro') ** 2
        loss_total = loss_view + gamma * loss_L2
        lo.append(loss_total)
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
            TG = np.sum(S * Y[v])
            Const_loss[v] += SE + alpha * BG + beta * TG

        Total = [1 / CL for CL in Const_loss]
        Total_sum = np.array(Total).sum()
        for v in range(V):
            omiga[v] = (Total[v]) / (Total_sum + eps)

        # Total = [CL for CL in Const_loss]
        # Total_sum = np.array(Total).sum()
        # for v in range(V):
        #     omiga[v] = (Total[v]) / (Total_sum + eps)
        # print(epoch)
        # print('*********************')
        # print(loss_total)
    print(lo)
    return S, omiga, B

# def Efficient_multi_view_clustering(H, B, alpha=1, beta=1, eps=1e-5, threshold=1e-5, epochs=100):
#     V = len(H)
#     Im = np.eye(B[0].shape[0])
#     omiga = [1 / V] * V
#     loss_last = 1e16
#     for epoch in range(epochs):
#
#         BBt = []
#         BHt = []
#         for v in range(V):
#             BBt_v = B[v].dot(B[v].T)
#             BHt_v = B[v].dot(H[v].T)
#
#             BBt.append(BBt_v)
#             BHt.append(BHt_v)
#
#         tmp1 = 0
#         tmp2 = 0
#         for v in range(V):
#             omiga_v_square = omiga[v] ** 2
#             tmp1 += omiga_v_square * (BBt[v] + alpha * Im)
#             tmp2 += omiga_v_square * BHt[v]
#         S = np.linalg.inv(tmp1 + beta * Im).dot(tmp2 * (1 + alpha))
#
#         loss_view = 0
#         for v in range(V):
#             loss_SE = np.linalg.norm(H[v].T - B[v].T.dot(S), 'fro') ** 2
#             loss_BG = np.linalg.norm(BHt[v] - S, 'fro') ** 2
#             omiga_v_square = omiga[v] ** 2
#             loss_view += omiga_v_square * (loss_SE + alpha * loss_BG)
#
#         loss_L2 = np.linalg.norm(S, 'fro') ** 2
#         loss_total = loss_view + beta * loss_L2
#         if loss_last - loss_total < threshold * loss_last:
#             break
#         else:
#             loss_last = loss_total
#         SSt = S.dot(S.T)
#         for v in range(V):
#             HtH_v = H[v].T.dot(H[v])
#             SH_v = S.dot(H[v])
#             B[v] = solve_sylvester(SSt, alpha * HtH_v, SH_v * (1 + alpha))
#
#         Const_loss = np.zeros(V)
#         for v in range(V):
#             SE = np.linalg.norm(H[v].T - B[v].T.dot(S), 'fro') ** 2
#             BG = np.linalg.norm(BHt[v] - S, 'fro') ** 2
#             Const_loss[v] += SE + alpha * BG
#
#         Total = [1 / CL for CL in Const_loss]
#         Total_sum = np.array(Total).sum()
#         for v in range(V):
#             omiga[v] = (Total[v]) / (Total_sum + eps)
#
#         # Total = [CL for CL in Const_loss]
#         # Total_sum = np.array(Total).sum()
#         # for v in range(V):
#         #     omiga[v] = (Total[v]) / (Total_sum + eps)
#
#     return S, omiga, B

# def Efficient_multi_view_clustering(H, B, alpha=1, beta=1, eps=1e-5, threshold=1e-5, epochs=100):
#     V = len(H)
#     Im = np.eye(B[0].shape[0])
#     omiga = [1 / V] * V
#     loss_last = 1e16
#     for epoch in range(epochs):
#
#         BBt = []
#         BHt = []
#         Y = []
#         for v in range(V):
#             BBt_v = B[v].dot(B[v].T)
#             BHt_v = B[v].dot(H[v].T)
#
#             BBt.append(BBt_v)
#             BHt.append(BHt_v)
#             B_sq = np.sum(B[v] ** 2, axis=1, keepdims=True)
#             H_sq = np.sum(H[v] ** 2, axis=1, keepdims=True).T
#             BH_dot = np.dot(B[v], H[v].T)
#             Y_v = B_sq + H_sq - 2 * BH_dot
#             Y.append(Y_v)
#
#
#         tmp1 = 0
#         tmp2 = 0
#         for v in range(V):
#             omiga_v_square = omiga[v] ** 2
#             tmp1 += omiga_v_square * BBt[v]
#             tmp2 += omiga_v_square * BHt[v]
#         S = np.linalg.inv(tmp1 + alpha * Im).dot(tmp2)
#
#         loss_view = 0
#         for v in range(V):
#             loss_SE = np.linalg.norm(H[v].T - B[v].T.dot(S), 'fro') ** 2
#             TG = np.sum(S * Y[v])
#             omiga_v_square = omiga[v] ** 2
#             loss_view += omiga_v_square * (loss_SE + alpha * TG)
#
#         loss_L2 = np.linalg.norm(S, 'fro') ** 2
#         loss_total = loss_view + beta * loss_L2
#         if loss_last - loss_total < threshold * loss_last:
#             break
#         else:
#             loss_last = loss_total
#         SSt = S.dot(S.T)
#         for v in range(V):
#             tem3 = np.linalg.inv(SSt)
#             SH_v = S.dot(H[v])
#             B[v] = tem3.dot(SH_v)
#
#         Const_loss = np.zeros(V)
#         for v in range(V):
#             SE = np.linalg.norm(H[v].T - B[v].T.dot(S), 'fro') ** 2
#             TG = np.sum(S * Y[v])
#             Const_loss[v] += SE +alpha * TG
#
#         Total = [1 / CL for CL in Const_loss]
#         Total_sum = np.array(Total).sum()
#         for v in range(V):
#             omiga[v] = (Total[v]) / (Total_sum + eps)
#
#         # Total = [CL for CL in Const_loss]
#         # Total_sum = np.array(Total).sum()
#         # for v in range(V):
#         #     omiga[v] = (Total[v]) / (Total_sum + eps)
#
#     return S, omiga, B

# def Efficient_multi_view_clustering(H, B, alpha=1, beta=1, gamma=1, eps=1e-5, threshold=1e-5, epochs=100):
#     V = len(H)
#     Im = np.eye(B[0].shape[0])
#     omiga = [1 / V] * V
#     loss_last = 1e16
#     for epoch in range(epochs):
#
#         BBt = []
#         BHt = []
#         Y = []
#         BBt_v = B[0].dot(B[0].T)
#         BBt.append(BBt_v)
#
#         for v in range(V):
#             BHt_v = B[0].dot(H[v].T)
#             BHt.append(BHt_v)
#
#             B_sq = np.sum(B[0] ** 2, axis=1, keepdims=True)
#             H_sq = np.sum(H[v] ** 2, axis=1, keepdims=True).T
#             BH_dot = np.dot(B[0], H[v].T)
#             Y_v = B_sq + H_sq - 2 * BH_dot
#             Y.append(Y_v)
#
#
#         tmp1 = 0
#         tmp2 = 0
#         for v in range(V):
#             omiga_v_square = omiga[v] ** 2
#             tmp1 += omiga_v_square * (BBt[0] + alpha * Im)
#             tmp2 += omiga_v_square * ((1+alpha)*BHt[v] - beta/2 * Y[v])
#
#         S = np.linalg.inv(tmp1 + gamma * Im).dot(tmp2)
#
#         loss_view = 0
#         for v in range(V):
#             loss_SE = np.linalg.norm(H[v].T - B[0].T.dot(S), 'fro') ** 2
#             loss_BG = np.linalg.norm(BHt[v] - S, 'fro') ** 2
#             loss_TG = np.sum(S * Y[v])
#             omiga_v_square = omiga[v] ** 2
#             loss_view += omiga_v_square * (loss_SE + alpha * loss_BG + beta * loss_TG)
#
#         loss_L2 = np.linalg.norm(S, 'fro') ** 2
#         loss_total = loss_view + gamma * loss_L2
#         if loss_last - loss_total < threshold * loss_last:
#             break
#         else:
#             loss_last = loss_total
#         SSt = S.dot(S.T)
#         for v in range(V):
#             HtH_v = H[v].T.dot(H[v])
#             SH_v = S.dot(H[v])
#             B[v] = solve_sylvester(SSt, alpha * HtH_v, SH_v * (1 + alpha))
#
#         Const_loss = np.zeros(V)
#         for v in range(V):
#             SE = np.linalg.norm(H[v].T - B[0].T.dot(S), 'fro') ** 2
#             BG = np.linalg.norm(BHt[v] - S, 'fro') ** 2
#             TG = np.sum(S * Y[v])
#             Const_loss[v] += SE + alpha * BG + beta * TG
#
#         Total = [1 / CL for CL in Const_loss]
#         Total_sum = np.array(Total).sum()
#         for v in range(V):
#             omiga[v] = (Total[v]) / (Total_sum + eps)
#
#     return S, omiga, B

# def Efficient_multi_view_clustering(H, B, alpha=1, beta=1, gamma=1, eps=1e-5, threshold=1e-5, epochs=100):
#     V = len(H)
#     Im = np.eye(B[0].shape[0])
#     omiga = [1 / V] * V
#     loss_last = 1e16
#     for epoch in range(epochs):
#
#         BBt = []
#         BHt = []
#         Y = []
#         S = []
#
#         for v in range(V):
#             BBt_v = B[v].dot(B[v].T)
#             BHt_v = B[v].dot(H[v].T)
#
#             BBt.append(BBt_v)
#             BHt.append(BHt_v)
#             B_sq = np.sum(B[v] ** 2, axis=1, keepdims=True)
#             H_sq = np.sum(H[v] ** 2, axis=1, keepdims=True).T
#             BH_dot = np.dot(B[v], H[v].T)
#             Y_v = B_sq + H_sq - 2 * BH_dot
#             Y.append(Y_v)
#
#
#         tmp1 = 0
#         tmp2 = 0
#         for v in range(V):
#             omiga_v_square = omiga[v] ** 2
#             tmp1 += omiga_v_square * (BBt[v] + alpha * Im)
#             tmp2 += omiga_v_square * ((1+alpha)*BHt[v] - beta/2 * Y[v])
#             S_v = np.linalg.inv(tmp1 + gamma * Im).dot(tmp2)
#             S.append(S_v)
#
#
#         loss_view = 0
#         for v in range(V):
#             loss_SE = np.linalg.norm(H[v].T - B[v].T.dot(S[v]), 'fro') ** 2
#             loss_BG = np.linalg.norm(BHt[v] - S[v], 'fro') ** 2
#             loss_TG = np.sum(S[v] * Y[v])
#             omiga_v_square = omiga[v] ** 2
#             loss_view += omiga_v_square * (loss_SE + alpha * loss_BG + beta * loss_TG)
#
#         loss_L2 = np.linalg.norm(S[v], 'fro') ** 2
#         loss_total = loss_view + gamma * loss_L2
#         if loss_last - loss_total < threshold * loss_last:
#             break
#         else:
#             loss_last = loss_total
#         for v in range(V):
#             SSt = S[v].dot(S[v].T)
#             HtH_v = H[v].T.dot(H[v])
#             SH_v = S[v].dot(H[v])
#             B[v] = solve_sylvester(SSt, alpha * HtH_v, SH_v * (1 + alpha))
#
#         Const_loss = np.zeros(V)
#         for v in range(V):
#             SE = np.linalg.norm(H[v].T - B[v].T.dot(S[v]), 'fro') ** 2
#             BG = np.linalg.norm(BHt[v] - S[v], 'fro') ** 2
#             TG = np.sum(S[v] * Y[v])
#             Const_loss[v] += SE + alpha * BG + beta * TG
#
#         Total = [1 / CL for CL in Const_loss]
#         Total_sum = np.array(Total).sum()
#         for v in range(V):
#             omiga[v] = (Total[v]) / (Total_sum + eps)
#
#     return S, omiga, B

# def Efficient_multi_view_clustering(H, B, alpha=1, beta=1, eps=1e-5, threshold=1e-5, epochs=100):
#     V = len(H)
#     Im = np.eye(B[0].shape[0])
#     loss_last = 1e16
#     for epoch in range(epochs):
#         BBt = []
#         BHt = []
#         S = []
#         for v in range(V):
#             BBt_v = B[v].dot(B[v].T)
#             BHt_v = B[v].dot(H[v].T)
#             BBt.append(BBt_v)
#             BHt.append(BHt_v)

#         for v in range(V):
#             tmp1 += BBt[v] + alpha * Im
#             S_v = np.linalg.inv(tmp1 + gamma * Im).dot(BHt[v])
#             S.append(S_v)
#
#         loss_view = 0
#         for v in range(V):
#             loss_SE = np.linalg.norm(H[v].T - B[v].T.dot(S[v]), 'fro') ** 2
#             loss_L2 = np.linalg.norm(S[v], 'fro') ** 2
#             loss_total = loss_SE + alpha * loss_L2
#         if loss_last - loss_total < threshold * loss_last:
#             break
#         else:
#             loss_last = loss_total
#         for v in range(V):
#             SSt = S[v].dot(S[v].T)
#             HtH_v = H[v].T.dot(H[v])
#             SH_v = S[v].dot(H[v])
#             B[v] = solve_sylvester(SSt, alpha * HtH_v, SH_v * (1 + alpha))
#
#         Const_loss = np.zeros(V)
#         for v in range(V):
#             SE = np.linalg.norm(H[v].T - B[v].T.dot(S[v]), 'fro') ** 2
#             BG = np.linalg.norm(BHt[v] - S[v], 'fro') ** 2
#             TG = np.sum(S[v] * Y[v])
#             Const_loss[v] += SE + alpha * BG + beta * TG
#
#         Total = [1 / CL for CL in Const_loss]
#         Total_sum = np.array(Total).sum()
#         for v in range(V):
#             omiga[v] = (Total[v]) / (Total_sum + eps)
#
#     return S, omiga, B

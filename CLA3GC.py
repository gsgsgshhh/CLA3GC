import numpy as np
from data_loader import datasets
from graph_filtering import multi_view_processing
from anchor import sampling_kmeans, sampling_minikmeans
from clustering import Efficient_multi_view_clustering, New_Efficient_clustering
from metrics import evaluate_clustering
# import matplotlib.pyplot as plt
import pandas as pd


import time

class clagc():

    # def __init__(self, dataname="Citeseer"):
    #     self.datasets = datasets
    #     self.alphas = [0.001, 1, 10, 100, 1000, 10000]
    #     self.betas = [0.001, 1, 10, 100, 1000, 10000]
    #     self.data_name = dataname
    #     self.data_type = 'graph data'
    #     self.max_dims = 100
    #     self.algorithms = [Efficient_clustering, Efficient_multi_view_clustering]
    #     self.k = 2
    #
    #     self.best_paras = {
    #         "alpha": 0,
    #         "beta": 0,
    #         "m": 0,
    #     }
    #     self.best_re = {
    #         "ACC": 0,
    #         "NMI": 0,
    #         "ARI": 0,
    #         "F1": 0,
    #         "PUR": 0,
    #         "Time": 0
    #     }

    def __init__(self, dataname="Citeseer"):
        self.datasets = datasets
        # self.alphas = [0.001]
        # self.betas = [10000]
        # self.gammas = [1]
        self.alphas = [0.001, 1, 10, 100, 1000, 10000]
        self.betas = [0.001, 1, 10, 100, 1000, 10000]
        self.gammas = [0.001, 1, 10, 100, 1000, 10000]   
        self.data_name = dataname
        self.data_type = 'graph data'
        self.max_dims = 200
        self.algorithms = [New_Efficient_clustering, Efficient_multi_view_clustering] 
        # self.algorithms = [Efficient_clustering, Efficient_multi_view_clustering]
        self.k = 2
        self.best_paras = {
            "alpha": 0,
            "beta": 0,
            "gamma": 0,
            "m": 0,
        }
        self.best_re = {
            "ACC": 0,
            "NMI": 0,
            "ARI": 0,
            "F1": 0,
            "PUR": 0,
            "Time": 0
        }

    def loadData(self):
        if self.data_name in ["ACM", "DBLP"]:
            X, As, Drs, gnd = self.datasets["multi-relational"](self.data_name)
            self.data_type = 'multi-relational '+ self.data_type
        elif self.data_name in ["Pubmed", "Citeseer", "Cora", "Products", "Papers100M", "ArXiv", "Mag"]:
            X, As, Drs, gnd = self.datasets["single-view"](self.data_name)
            self.data_type = 'single-view ' + self.data_type
        elif self.data_name in ["AMAP", "AMAC"]:
            X, As, Drs, gnd = self.datasets["multi-attribute"](self.data_name)
            X, As, Drs, gnd = self.datasets["multi-attribute"](self.data_name)
        else:
            print("No such dataset!!!!!")
            return

        self.gnd = gnd
        self.C = len(np.unique(self.gnd))
        # self.ms = [self.C, 100]
        self.ms = [self.C, 10, 30, 50, 70, 100]
        self.N = X[0].shape[0]
        return X, As, Drs, gnd

    def graphFiltering(self, k = 2):
        X, As, Drs, _ = self.loadData()
        H = multi_view_processing(X=X, A=As, Dr=Drs, k=k, dims=self.max_dims)
        self.V = len(H)
        return H

    # def initailizeB(self, m, H):
    #     B = []
    #     for v in range(self.V):
    #         if self.N > 30000:
    #             B_tmp = sampling_minikmeans(H, m)[0]
    #             B.append(B_tmp)
    #         else:
    #             B_tmp = sampling_kmeans(H, m)[0]
    #             B.append(B_tmp)
    #     return B

    def initailizeB(self, m, H):
        B = []
        if self.N > 30000:
            B_tmp = sampling_minikmeans(H, m)[0]
            B.append(B_tmp)
        else:
            B_tmp = sampling_kmeans(H, m)

        return B_tmp

    def showData(self):
        print("------------------------------------------------------------------------------")
        print("{} is {} with {} view(s) and {} nodes".format(self.data_name, self.data_type, self.V, self.N))
        print("------------------------------------------------------------------------------")

    # def train(self):
    #     H = self.graphFiltering()
    #     self.showData()
    #     ti = 0
    #     total_time = 0
    #     for m in self.ms:
    #         if m < self.C:
    #             continue
    #         B = self.initailizeB(m=m, H=H)
    #
    #         for alpha in self.alphas:
    #             for beta in self.betas:
    #                 time_begin = time.time()
    #                 if self.V == 1:
    #                     Z, _ = self.algorithms[0](H[0], B[0], alpha=alpha, beta=beta)
    #                 else:
    #                     Z, _, _ = self.algorithms[1](H, B, alpha=alpha, beta=beta)
    #                 time_end = time.time()
    #                 Time = np.fabs(time_end - time_begin)
    #                 ti += 1
    #                 total_time += Time
    #                 acc, nmi, ari, f1, pur = evaluate_clustering(Z, self.gnd)
    #                 print(
    #                     "m: {0: <3} alpha: {1: <5} beta: {2: <5} ACC: {3:.4f} NMI: {4:.4f} ARI: {5:.4f} F1: {6:.4f} PUR: {7:.4f} Time: {8:.4f}".format(
    #                         m, alpha, beta, acc, nmi, ari, f1, pur, Time
    #                     )
    #                 )
    #                 if acc > self.best_re["ACC"]:
    #                     self.best_re["ACC"] = acc
    #                     self.best_re["NMI"] = nmi
    #                     self.best_re["ARI"] = ari
    #                     self.best_re["F1"] = f1
    #                     self.best_re["PUR"] = pur
    #                     self.best_re["Time"] = Time
    #                     self.best_paras["alpha"] = alpha
    #                     self.best_paras["beta"] = beta
    #                     self.best_paras["m"] = m
    #     print(
    #         "k: {0: <3} m: {1: <3} alpha: {2: <5} beta: {3: <5} ACC: {4:.4f} NMI: {5:.4f} ARI: {6:.4f} F1: {7:.4f} PUR: {8:.4f} Time: {9:.2f}".format(
    #                 self.k, self.best_paras["m"], self.best_paras["alpha"], self.best_paras["beta"], self.best_re["ACC"], self.best_re["NMI"], self.best_re["ARI"], self.best_re["F1"], self.best_re["PUR"],  self.best_re["Time"]
    #                 )
    #         )

    def train(self):
        H = self.graphFiltering()
        self.showData()
        ti = 0
        total_time = 0
        for m in self.ms:
            if m < self.C:
                continue
            B = self.initailizeB(m=m, H=H)

            for alpha in self.alphas:
                for beta in self.betas:
                    for gamma in self.gammas:
                        time_begin = time.time()
                        if self.V == 1:
                            Z, _ = self.algorithms[0](H[0], B[0], alpha=alpha, beta=beta, gamma=gamma)
                        else:
                            Z, _, _ = self.algorithms[1](H, B, alpha=alpha, beta=beta, gamma=gamma)
                        time_end = time.time()
                        Time = np.fabs(time_end - time_begin)
                        ti += 1
                        total_time += Time
                        acc, nmi, ari, f1, pur = evaluate_clustering(Z, self.gnd)
                        print(
                            "m: {0: <3} alpha: {1: <5} beta: {2: <5} gamma: {3: <5} ACC: {4:.4f} NMI: {5:.4f} ARI: {6:.4f} F1: {7:.4f} PUR: {8:.4f} Time: {9:.4f}".format(
                                m, alpha, beta, gamma, acc, nmi, ari, f1, pur, Time
                            )
                        )
                        if acc > self.best_re["ACC"]:
                            self.best_re["ACC"] = acc
                            self.best_re["NMI"] = nmi
                            self.best_re["ARI"] = ari
                            self.best_re["F1"] = f1
                            self.best_re["PUR"] = pur
                            self.best_re["Time"] = Time
                            self.best_paras["alpha"] = alpha
                            self.best_paras["beta"] = beta
                            self.best_paras["gamma"] = gamma
                            self.best_paras["m"] = m
        print(
            "k: {0: <3} m: {1: <3} alpha: {2: <5} beta: {3: <5} gamma: {4: <5} ACC: {5:.4f} NMI: {6:.4f} ARI: {7:.4f} F1: {8:.4f} PUR: {9:.4f} Time: {10:.2f}".format(
                    self.k, self.best_paras["m"], self.best_paras["alpha"], self.best_paras["beta"], self.best_paras["gamma"], self.best_re["ACC"], self.best_re["NMI"], self.best_re["ARI"], self.best_re["F1"], self.best_re["PUR"],  self.best_re["Time"]
                    )
            )


if __name__ == "__main__":
    dataname = "AMAP"
    res = clagc(dataname=dataname)
    res.train()












from data_preprocess import load_dataset
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch
from torch_geometric.datasets import Planetoid
from collections import Counter
from ogb.nodeproppred import PygNodePropPredDataset

def single_view_graphs(dataname='Pubmed'):
    file_root = './data'
    if dataname == "Products":
        dataname = "ogbn-products"
        file_root = './data/ogb'
    if dataname == "ArXiv":
        dataname = "ogbn-arxiv"
        file_root = './data/ogb'
    if dataname == "ogbn-papers100M":
        dataname = "ogbn-papers100M"
        file_root = './data/ogb'
    large_graph_dataset = {
        "Pubmed": Planetoid,
        "Citeseer": Planetoid,
        "Cora": Planetoid,
        "ogbn-products": PygNodePropPredDataset,
        "ogbn-arxiv": PygNodePropPredDataset,
        "ogbn-papers100M": PygNodePropPredDataset

    }
    dataset = large_graph_dataset["{}".format(dataname)](root='{}/{}'.format(file_root, dataname.lower()),name=dataname)
    data = dataset[0]
    X = data.x.numpy()
    N = X.shape[0]
    gnd = data.y.numpy()
    row = data.edge_index[0].numpy()
    col = data.edge_index[1].numpy()
    degree = np.zeros(N)
    Ct = Counter(row)
    degree[list(Ct.keys())] = list(Ct.values())
    M = data.num_edges
    values = torch.ones(M)
    adj = csr_matrix((values, (row, col)), shape=(N, N))
    X = [X]
    adj = [adj]
    degree = [degree]
    if "ogbn" in dataname:
        gnd = gnd[:, 0]
    return X, adj, degree, gnd

def multi_attribute_graphs(dataname="AMAP"):
    Amazon = {
        "AMAP":'amazon_photos',
        "AMAC":'amazon_computers'
    }
    dataname = Amazon["{}".format(dataname)]
    Amazon = load_dataset("./data/npz/{}.npz".format(dataname))
    Adj = sp.csr_matrix(Amazon.standardize().adj_matrix).A
    Attr = sp.csr_matrix(Amazon.standardize().attr_matrix).A
    Gnd = sp.csr_matrix(Amazon.standardize().labels).A
    Gnd = Gnd.T.squeeze()
    X = []
    Attr = np.array(Attr)
    X.append(Attr)
    X.append(Attr.dot(Attr.T))
    A = np.array(Adj)
    Dr = np.sum(A, axis=1)
    Dr = [Dr]
    A = csr_matrix(A)
    A = [A]
    return X, A, Dr, Gnd

def multi_relational_graphs(dataname="ACM"):
    if dataname == "ACM" :
        X, Adj, Gnd = Acm()
    elif dataname == "DBLP" :
        X, Adj, Gnd = Dblp()
    else:
        print("No such dataset")
        return None, None, None, None
    As = []
    Drs = []
    for A in Adj:
        Dr = A.sum(axis=1)
        Drs.append(Dr)
        A = csr_matrix(A)
        As.append(A)
    X = [X]
    return X, As, Drs, Gnd

def Acm():
    dataset = "./data/mat/" + 'ACM'
    data = sio.loadmat('{}.mat'.format(dataset))
    X = data['features']
    A = data['PAP']
    B = data['PLP']
    if sp.issparse(X) :
        X = X.todense()
    As = []
    A = A.toarray()
    B = B.toarray()
    X = np.array(X)
    As.append(A)
    As.append(B)
    gnd = data['label']
    gnd = gnd.T
    gnd = np.argmax(gnd, axis=0)
    return X, As, gnd

def Dblp() :
    dataset = "./data/mat/" + 'DBLP'
    data = sio.loadmat('{}.mat'.format(dataset))
    X = data['features']
    A = data['net_APTPA']
    B = data['net_APCPA']
    C = data['net_APA']
    if sp.issparse(X) :
        X = X.todense()
    As = []
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    As.append(A)
    As.append(B)
    As.append(C)
    X = np.array(X)
    gnd = data['label']
    gnd = gnd.T
    gnd = np.argmax(gnd, axis=0)
    return X, As, gnd

datasets = {
    "single-view": single_view_graphs,
    "multi-relational": multi_relational_graphs,
    "multi-attribute": multi_attribute_graphs,

}

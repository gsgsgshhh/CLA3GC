from data_preprocess import load_dataset
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch
from torch_geometric.datasets import Planetoid
from collections import Counter
from ogb.nodeproppred import PygNodePropPredDataset

# 加载单视图数据集
def single_view_graphs(dataname='Pubmed'):

    # 数据集路径设置
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

    # 数据集字典（设置对应的加载类）
    large_graph_dataset = {
        "Pubmed": Planetoid,
        "Citeseer": Planetoid,
        "Cora": Planetoid,
        "ogbn-products": PygNodePropPredDataset,
        "ogbn-arxiv": PygNodePropPredDataset,
        "ogbn-papers100M": PygNodePropPredDataset

    }

    # 加载数据集，转换为图数据
    dataset = large_graph_dataset["{}".format(dataname)](root='{}/{}'.format(file_root, dataname.lower()),name=dataname)
    data = dataset[0]
    X = data.x.numpy() # 提取特征矩阵
    N = X.shape[0]  # 获取图中节点的数量
    gnd = data.y.numpy() # 获取图中每个节点的标签

    # 计算节点的度
    row = data.edge_index[0].numpy()
    col = data.edge_index[1].numpy()
    degree = np.zeros(N)
    Ct = Counter(row)
    degree[list(Ct.keys())] = list(Ct.values()) # 将每个节点的度存储在degree数组中

    # 构建邻接矩阵
    M = data.num_edges
    values = torch.ones(M)
    adj = csr_matrix((values, (row, col)), shape=(N, N)) # 压缩行存储格式，构建稀疏的邻接矩阵

    # 将特征矩阵、邻接矩阵、度都转换为列表（后续使用更方便）
    X = [X]
    adj = [adj]
    degree = [degree]

    #对于ogbn数据集的标签需要特殊处理（标签是多维的，只需要第一个标签列）
    if "ogbn" in dataname:
        gnd = gnd[:, 0]

    return X, adj, degree, gnd


# 具有多重属性的多视图数据集
def multi_attribute_graphs(dataname="AMAP"):

    # 选择并加载数据集
    Amazon = {
        "AMAP":'amazon_photos',
        "AMAC":'amazon_computers'
    }
    dataname = Amazon["{}".format(dataname)]
    Amazon = load_dataset("./data/npz/{}.npz".format(dataname))

    # 预处理（标准化邻接矩阵和属性矩阵并转化为稀疏矩阵存储）
    Adj = sp.csr_matrix(Amazon.standardize().adj_matrix).A
    Attr = sp.csr_matrix(Amazon.standardize().attr_matrix).A
    Gnd = sp.csr_matrix(Amazon.standardize().labels).A
    Gnd = Gnd.T.squeeze()
    X = []
    Attr = np.array(Attr)
    X.append(Attr)
    X.append(Attr.dot(Attr.T))

    # 计算节点度
    A = np.array(Adj)
    Dr = np.sum(A, axis=1)
    Dr = [Dr]

    # 创建稀疏邻接矩阵
    A = csr_matrix(A)
    A = [A]

    return X, A, Dr, Gnd


# 具有多重拓扑关系的多视图数据集
def multi_relational_graphs(dataname="ACM"):

    # 论文只用到这两个数据集
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


def Acm() :
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

# 数据集字典（不同类型的数据集对应调用不同的函数）
datasets = {
    "single-view": single_view_graphs,
    "multi-relational": multi_relational_graphs,
    "multi-attribute": multi_attribute_graphs,
}
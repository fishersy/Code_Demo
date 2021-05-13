# -*- utf-8 -*-
# author : fisherwsy
import numpy as np
from tqdm import tqdm
from itertools import product
import networkx as nx
N,T,maxlag = 15,1000,3


def generate_data_linear_DAG(N,T,maxlag):

    # 噪声
    data = np.random.randn(T,N)
    # 生成邻接矩阵
    edge_mat = np.triu((np.random.uniform(0,1,[N,N])<0.3).astype('int'),1)  # 这里可以不用是上三角的
    weight_mat = np.random.uniform(0.3,1,[N,N]) * edge_mat  # 这里的权重用的都是正数,也可以是负数不过在这里就无所谓了
    lag_mat = np.random.randint(1,maxlag+1,[N,N]) * edge_mat

    for t in tqdm(range(maxlag,len(data))):
        for i in range(N):
            for j in range(i):
                data[t,i] += data[t-lag_mat[j,i],j] * weight_mat[j,i]
    return data,edge_mat,weight_mat,lag_mat



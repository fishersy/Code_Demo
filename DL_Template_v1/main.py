# -*- utf-8 -*-
# author : fisherwsy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import torch.optim as optim
from models import LSTM_base,Linear_base,Linear_base_1,Linear_base_3,lstm_base,Dense_base,MLP_base,CNN_base
from utils import generate_data_linear_DAG,generate_data_linear_DAG_2
from tqdm import tqdm
import pickle
from torch.utils.tensorboard import SummaryWriter
import logging
from utils import get_device
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    N = 15
    maxlag = 3
    T = 3000
    data,edge_mat,weight_mat,lag_mat = generate_data_linear_DAG_2(N,T,maxlag)
    data = data.astype('float32')
    # sns.heatmap(lag_mat, square=True)
    # plt.savefig('lag_mat.png')
    # plt.clf()
    sns.heatmap(weight_mat, square=True)
    plt.savefig('weight_mat.png')
    plt.clf()

    device = get_device()
    lstm_output_dim = 10
    Lstm_nums_layers = 3

    model = CNN_base(N, lstm_output_dim, Lstm_nums_layers, device = device).to(device)
    lr = 0.01
    optimizer = optim.Adam(model.parameters(),lr=lr)
    batch_size = 100
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    for epoch in range(200):
        model.train()
        for i, Input in enumerate(dataloader):
            # fowards

            loss,l1 = model(Input.to(device))
            Loss = loss + l1*0.01
            # Loss = loss
            # Backward
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
            adj = model.get_adj()
        # recall,precision,f1 = get_performance(adj,edge_mat,0.5)
        # print(recall,precision,f1,L1, L2, loss)
        print(loss,roc_auc_score(edge_mat.flatten(),adj.flatten()))
        sns.heatmap(adj, square=True)
        plt.savefig('adj.png')
        plt.clf()
        pickle.dump(adj,open('adj.pkl','wb'))




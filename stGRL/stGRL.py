import torch
from preprocess import preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, \
    construct_interaction_KNN, add_contrastive_label, get_feature, permutation, fix_seed
import time
import random
import numpy as np
import sys

sys.path.append('/home/luxin1/stGRL')
from model import Encoder, Encoder_sparse,  ZINBLoss, GaussianNoise, MeanAct, DispAct
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
import matplotlib.pyplot as plt


class stGRL():
    def __init__(self,
                 adata,
                 device=torch.device('cuda'),
                 learning_rate=0.001,
                 learning_rate_sc=0.01,
                 weight_decay=0.00,
                 epochs=500,
                 n_top_genes=5000,
                 dim_output=256,
                 random_seed=41,
                 alpha=10,
                 beta=0.5,
                 gama=0.5,
                 save_reconstruction=False,
                 datatype='10X',
                 n_neighbor=3,
                 ):
        self.adata = adata.copy()
        self.device = device
        self.learning_rate = learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.n_top_genes = n_top_genes
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.datatype = datatype
        self.save_reconstruction = save_reconstruction
        self.n_neighbor = n_neighbor

        fix_seed(self.random_seed)

        if 'highly_variable' not in adata.var.keys():
            preprocess(self.adata, self.n_top_genes)

        if 'adj' not in adata.obsm.keys():
            if self.datatype in ['Stereo', 'Slide']:
                construct_interaction_KNN(self.adata,self.n_neighbor)
            else:
                construct_interaction(self.adata,self.n_neighbor)

        if 'feat' not in adata.obsm.keys():
            get_feature(self.adata)

        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(
            self.device)

        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output

        if self.datatype in ['Stereo', 'Slide']:
            # using sparse
            print('Building sparse matrix ...')
            self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else:
            # standard version
            self.adj = preprocess_adj(self.adj)
            self.adj = torch.FloatTensor(self.adj).to(self.device)


    def train(self):
        if self.datatype in ['Stereo', 'Slide']:
            self.model = Encoder_sparse(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        else:
            self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay)

        print('Begin to train ST data...')
        self.model.train()
        copyemb = None
        loss_list = []
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            self.features_a = permutation(self.features)

            self.hiden_feat, self.emb,  zinb_loss, meanbatch, dispbatch, pibatch ,self.dec,self.dec_a= self.model(self.features,
                                                                                                                  self.features_a,
                                                                                                                  self.adj)
            if epoch == 0:
                copyemb = self.emb

            self.loss_feat = F.mse_loss(self.features, self.emb)
            zinb_loss = zinb_loss(self.features, meanbatch, dispbatch, pibatch, device=self.device)
            semi_loss = self.model.contraction_loss(self.dec, self.dec_a)

            loss = self.alpha * self.loss_feat + self.beta * zinb_loss + self.gama * semi_loss
            nan_count = torch.isnan(self.emb).sum()
            print(nan_count.item())
            if nan_count.item() > 0:
                self.emb = copyemb
                break

            copyemb = self.emb
            loss_list.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        plt.figure(figsize=(10, 5))
        plt.plot(loss_list, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.show()


        print("Optimization finished for ST data!")

        with torch.no_grad():
            self.model.eval()
            self.adata.obsm['rec'] = self.emb.detach().cpu().numpy()
            self.emb = F.normalize(self.emb, p=2, dim=1).detach().cpu().numpy()
            self.adata.obsm['emb'] = self.emb
            return self.adata
    








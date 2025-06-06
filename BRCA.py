import sys
sys.path.append('/home/luxin1/stGRL-main/stGRL/')
from stGRL import stGRL
import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['R_HOME'] = '/home/luxin1/miniconda3/envs/pytorch/lib/R'

file_fold = "/home/luxin1/ST-DATA/Human_breast_cancer_data/"
n_clusters = 20

adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
adata.var_names_make_unique()

model = stGRL(adata,device = device,n_top_genes=4000,dim_output=128,n_neighbor=4)
adata = model.train()

tool = 'mclust' # mclust, leiden, and louvain
# clustering
from utils import clustering
radius= 50
clustering(adata, n_clusters, radius=radius, method=tool, refinement=True)

df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
df_meta_layer = df_meta['fine_annot_type']
df_meta['stGRL'] = adata.obs['stGRL'].tolist()
df_meta.to_csv(f'{file_fold}/metadata.tsv', index=False, sep='\t')
adata.obs['ground_truth'] = df_meta_layer.values
adata = adata[~pd.isnull(adata.obs['ground_truth'])]

    # calculate metric ARI
ARI = metrics.adjusted_rand_score(adata.obs['stGRL'], adata.obs['ground_truth'])
NMI = metrics.normalized_mutual_info_score(adata.obs['stGRL'], adata.obs['ground_truth'])
AMI = metrics.adjusted_mutual_info_score(adata.obs['stGRL'], adata.obs['ground_truth'])
print(ARI, NMI, AMI)
sc.pl.spatial(adata,
                    img_key="hires",
                    color=[ "stGRL","stGRL","stGRL"],
                    title=["ARI=%.4f(stGRL)" % ARI, "NMI=%.4f(stGRL)" % NMI,
                     "AMI=%.4f(stGRL)" % AMI],
                    show=False, save=False)







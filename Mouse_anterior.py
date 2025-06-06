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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
os.environ['R_HOME'] = '/home/luxin1/miniconda3/envs/pytorch/lib/R'
n_clusters = 52
file_fold = '/home/luxin1/stGRL-main/Mouse_anterior_brain/'
adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)

adata.var_names_make_unique()

model = stGRL(adata, device=device, n_top_genes=4000, dim_output=256, n_neighbor=3)
adata = model.train()
from utils import clustering
radius = 50
tool = 'mclust'
clustering(adata, n_clusters, radius=radius, method=tool, refinement=True)

df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
de_meta_layer = df_meta['ground_truth']
adata.obs['ground_truth'] = de_meta_layer.values
adata = adata[~pd.isnull(adata.obs['ground_truth'])]
ARI = metrics.adjusted_rand_score(adata.obs['stGRL'], adata.obs['ground_truth'])
NMI = metrics.normalized_mutual_info_score(adata.obs['stGRL'], adata.obs['ground_truth'])
AMI = metrics.adjusted_mutual_info_score(adata.obs['stGRL'], adata.obs['ground_truth'])
print(ARI, NMI, AMI)

import matplotlib.pyplot as plt
sc.pl.spatial(adata,
                   img_key="hires",
                   color=["stGRL","stGRL","stGRL"],
                   title=["ARI=%.4f(stGRL)" % ARI, "NMI=%.4f(stGRL)" % NMI,
                     "AMI=%.4f(stGRL)" % AMI],
                   show=False,save=False)
plt.savefig('/home/luxin1/stGRL-main/Mouse/stGRL-ALL.pdf', format='pdf', bbox_inches='tight',transparent=True)
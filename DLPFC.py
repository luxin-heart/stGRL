import sys
sys.path.append('your path')
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
sample_list = ['151507', '151508', '151509', '151510',
               '151669', '151670', '151671', '151672',
               '151673', '151674', '151675', '151676']
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
os.environ['R_HOME'] = 'your path'
save_path = 'your path'
os.makedirs(save_path, exist_ok=True)
ARI_list = []
NMI_list = []
AMI_list = []


for sample_name in sample_list:
    file_fold = '' + str(sample_name)
    
    if sample_name in ['151669', '151670', '151671', '151672']:
        n_clusters = 5
    else:
        n_clusters = 7

    adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    model = stGRL(adata, device=device, n_top_genes=4000, dim_output=256)
    adata = model.train()
    print(sample_name)
    from utils import clustering

    tool = 'mclust'
    radius = 50
    clustering(adata, n_clusters, radius=radius, method=tool, refinement=True)
    df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
    # save result
    # df_meta['stGRL'] = adata.obs['stGRL'].tolist()
    # df_meta.to_csv(f'{file_fold}/metadata.tsv', index=False, sep='\t')
    de_meta_layer = df_meta['layer_guess']
    adata.obs['ground_truth'] = de_meta_layer.values
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    ARI = metrics.adjusted_rand_score(adata.obs['stGRL'], adata.obs['ground_truth'])
    adata.uns['ARI'] = ARI
    ARI_list.append(ARI)
    NMI = metrics.normalized_mutual_info_score(adata.obs['stGRL'], adata.obs['ground_truth'])
    NMI_list.append(NMI)
    AMI = metrics.adjusted_mutual_info_score(adata.obs['stGRL'], adata.obs['ground_truth'])
    AMI_list.append(AMI)
    sc.pl.spatial(adata,
                   img_key="hires",
                   color=["ground_truth", "stGRL", "stGRL", "stGRL"],
                   title=["Ground truth", "ARI=%.4f(stGRL)" % ARI, "NMI=%.4f(stGRL)" % NMI, "AMI=%.4f(stGRL)" % AMI],
                   show=True, save=False)
    plt.savefig(os.path.join(save_path, f'{sample_name}.pdf'), dpi=600)
    plt.close()
print(ARI_list, np.mean(ARI_list))
print(NMI_list, np.mean(NMI_list))
print(AMI_list, np.mean(AMI_list))

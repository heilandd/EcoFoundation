# run_c2l_script.py

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cell2location
from matplotlib import rcParams
import pandas as pd
import os
import argparse

def runC2L(adata_path, infer_csv, output, maxepochs):
    
    adata = sc.read(adata_path)
    
    if os.path.exists(output):
        print(f"'{output}' exists. No further analysis")
    else:
        print(f"'{output}' does not exist. Run C2L")
        adata_vis = adata
        adata_vis.var_names_make_unique()
        adata_vis.var['SYMBOL'] = adata_vis.var_names
        adata_vis.var['MT_gene'] = [gene.startswith('MT-') for gene in adata_vis.var['SYMBOL']]
        adata_vis.obsm['MT'] = adata_vis[:, adata_vis.var['MT_gene'].values].X.toarray()

        inf_aver = pd.read_csv(infer_csv, index_col=0)
        intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
        adata_vis = adata_vis[:, intersect].copy()
        inf_aver = inf_aver.loc[intersect, :].copy()

        cell2location.models.Cell2location.setup_anndata(adata=adata_vis)
        mod = cell2location.models.Cell2location(
            adata_vis, cell_state_df=inf_aver,
            N_cells_per_location=8, detection_alpha=20
        )

        mod.train(max_epochs=int(maxepochs), batch_size=None, train_size=1)
        adata_vis = mod.export_posterior(
            adata_vis,
            sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs}
        )

        cell_types = adata_vis.obsm["q05_cell_abundance_w_sf"]
        cell_types.to_csv(output)
        adata_vis.write(adata_path)
        return cell_types


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cell2location model")
    parser.add_argument('--adata_path', required=True, help='Path to input AnnData file (h5ad)')
    parser.add_argument('--infer_csv', required=True, help='Path to inferred average expression (CSV)')
    parser.add_argument('--output', required=True, help='Path to save Cell2Location output (CSV)')
    parser.add_argument('--maxepochs', default=250, help='Number of training epochs')

    args = parser.parse_args()

    # Load adata
    #adata = sc.read_h5ad(args.adata)

    # Run C2L
    runC2L(args.adata_path, args.infer_csv, args.output, args.maxepochs)

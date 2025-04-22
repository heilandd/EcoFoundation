# run_cell2location_sc.py

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cell2location
from matplotlib import rcParams
from cell2location.utils.filtering import filter_genes
from cell2location.models import RegressionModel
import argparse
import os

rcParams['pdf.fonttype'] = 42  # enables correct plotting of text for PDFs


def runCell2LocationSC(results_folder, adata_path):
    ref_run_name = f'{results_folder}/reference_signatures'
    run_name = f'{results_folder}/cell2location_map'
    model_file = os.path.join(ref_run_name, "model.pt")  # model.pt is the saved model file

    # Check if model already exists
    if os.path.exists(model_file):
        print(f"✅ Model already exists at: {model_file}. Skipping training.")
        adata_ref = sc.read(adata_path)
        adata_ref.var['SYMBOL'] = adata_ref.var.index
        #adata_ref.var.set_index('GeneID-2', drop=True, inplace=True)
        if hasattr(adata_ref, "raw"):
            del adata_ref.raw

        selected = filter_genes(
            adata_ref, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)
        adata_ref = adata_ref[:, selected].copy()
        
        cell2location.models.RegressionModel.setup_anndata(
            adata=adata_ref,
            batch_key='batch')
        mod = RegressionModel(adata_ref)
        
    else:
        print(f"🚀 Training model. Output will be saved to: {ref_run_name}")
        adata_ref = sc.read(adata_path)
        adata_ref.var['SYMBOL'] = adata_ref.var.index
        #adata_ref.var.set_index('GeneID-2', drop=True, inplace=True)
        if hasattr(adata_ref, "raw"):
            del adata_ref.raw

        selected = filter_genes(
            adata_ref, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)
        adata_ref = adata_ref[:, selected].copy()
        
        cell2location.models.RegressionModel.setup_anndata(
            adata=adata_ref,
            batch_key='batch',
            labels_key='celltype')
        mod = RegressionModel(adata_ref)
        mod.train(max_epochs=200)
        adata_ref = mod.export_posterior(adata_ref, sample_kwargs={'num_samples': 1000, 'batch_size': 2500})
        mod.save(f"{ref_run_name}", overwrite=True)
        adata_file = f"{ref_run_name}/sc.h5ad"
        adata_ref.write(adata_file)

    ## Run with pretrained model:
    adata_file = f"{ref_run_name}/sc.h5ad"
    adata_ref = sc.read(adata_file)
    mod = cell2location.models.RegressionModel.load(f"{ref_run_name}", adata_ref)
    
    adata_ref = mod.export_posterior(
        adata_ref,
        use_quantiles=True,
        add_to_varm=["q05","q50", "q95", "q0001"],
        sample_kwargs={'batch_size': 2500}
    )
    
    adata_file = f"{ref_run_name}/sc_mods.h5ad"
    adata_ref.write(adata_file)
    
    if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
        inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                        for i in adata_ref.uns['mod']['factor_names']]].copy()
    else:
        inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}'
                                        for i in adata_ref.uns['mod']['factor_names']]].copy()


    inf_aver.columns = adata_ref.uns['mod']['factor_names']
    inf_aver.to_csv(f"{ref_run_name}/infer.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cell2Location RegressionModel")
    parser.add_argument('--results_folder', required=True, help='Output folder for model results')
    parser.add_argument('--adata_path', required=True, help='Input path to scRNA-seq .h5ad file')
    args = parser.parse_args()

    runCell2LocationSC(args.results_folder, args.adata_path)

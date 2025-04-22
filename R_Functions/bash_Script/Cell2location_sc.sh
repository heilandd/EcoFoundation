#!/bin/bash

# =============================================================================
# Script: run_cell2location_sc.sh
# Description: Runs Cell2Location's RegressionModel on a given scRNA-seq dataset
#              inside a specified conda environment ("c2l").
#
# Usage:
#   ./Cell2Location_sc.sh <RESULTS_FOLDER> <ADATA_PATH>
#
# Arguments:
#   RESULTS_FOLDER   Path to the output directory where results will be saved
#   ADATA_PATH       Path to the input .h5ad file containing scRNA-seq data
#
# Example:
#   ./run_cell2location_sc.sh ./results ./data/sample.h5ad
#
# Requirements:
#   - Conda environment 'c2l' must be installed and configured
#   - Python script 'run_cell2location_sc.py' must be in the same directory
# =============================================================================

# --------------------
# Help option
# --------------------
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  sed -n '2,20p' "$0" | sed 's/^#//'
  exit 0
fi

# --------------------
# Argument check
# --------------------
if [ "$#" -ne 2 ]; then
  echo "❌ Error: Invalid number of arguments."
  echo "Run with -h for help."
  exit 1
fi

# --------------------
# Assign input arguments
# --------------------
RESULTS_FOLDER="$1"
ADATA_PATH="$2"

# --------------------
# Create results folder if it does not exist
# --------------------
if [ ! -d "$RESULTS_FOLDER" ]; then
  echo "📁 Creating results directory: $RESULTS_FOLDER"
  mkdir -p "$RESULTS_FOLDER"
else
  echo "📁 Using existing results directory: $RESULTS_FOLDER"
fi

# --------------------
# Check if input file exists
# --------------------
if [ ! -f "$ADATA_PATH" ]; then
  echo "❌ Error: Input file '$ADATA_PATH' does not exist."
  exit 1
fi

# --------------------
# Activate conda and run script
# --------------------
echo "✅ Activating conda environment: c2l"
source /Users/henrikheiland/miniconda/etc/profile.d/conda.sh
conda activate c2l

echo "🚀 Running Cell2Location RegressionModel..."
python /Users/henrikheiland/Desktop/MERFISH/EcoFoundation/R_Functions/bash_Script/run_cell2location_sc.py --results_folder "$RESULTS_FOLDER" --adata_path "$ADATA_PATH"

echo "✅ Finished. Results saved in: $RESULTS_FOLDER"

# --------------------
# Deactivate conda
# --------------------
conda deactivate
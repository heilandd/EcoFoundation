#!/bin/bash

# =============================================================================
# Script: run_c2l.sh
# Description: Runs Cell2Location's model (C2L) using a reference CSV and 
#              spatial data, inside the conda environment "c2l".
#
# Usage:
#   ./run_c2l.sh <AD_FILE> <INF_CSV> <OUTPUT_CSV> [MAXEPOCHS]
#
# Arguments:
#   AD_FILE      Path to input spatial .h5ad file
#   INF_CSV      Path to inferred average expression CSV (from scRNA model)
#   OUTPUT_CSV   Output path to save Cell2Location spatial mapping results
#   MAXEPOCHS    (Optional) Number of training epochs [default: 300]
#
# Example:
#   ./run_c2l.sh ./data/spatial.h5ad ./results/reference/infer.csv ./results/spatial_map.csv 250
#
# Help:
#   ./run_c2l.sh --help
#
# Requirements:
#   - Conda environment 'c2l'
#   - Python script 'run_c2l_script.py' in the same directory
# =============================================================================

# ----------------------------
# Show help message
# ----------------------------
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  sed -n '2,25p' "$0" | sed 's/^#//'
  exit 0
fi

# ----------------------------
# Validate inputs
# ----------------------------
if [ "$#" -lt 3 ]; then
  echo "❌ Error: Missing arguments."
  echo "Usage: ./run_c2l.sh <AD_FILE> <INF_CSV> <OUTPUT_CSV> [MAXEPOCHS]"
  exit 1
fi

# ----------------------------
# Read input arguments
# ----------------------------
AD_FILE="$1"
INF_CSV="$2"
OUTPUT_CSV="$3"
MAXEPOCHS="${4:-300}"  # default to 300 if not provided

# ----------------------------
# Validate input paths
# ----------------------------
if [ ! -f "$AD_FILE" ]; then
  echo "❌ Error: AD_FILE '$AD_FILE' not found."
  exit 1
fi

if [ ! -f "$INF_CSV" ]; then
  echo "❌ Error: INF_CSV '$INF_CSV' not found."
  exit 1
fi

# ----------------------------
# Activate conda environment
# ----------------------------
echo "✅ Activating conda environment: c2l"
source /Users/henrikheiland/miniconda/etc/profile.d/conda.sh
conda activate c2l

# ----------------------------
# Run Python script
# ----------------------------
echo "🚀 Running Cell2Location..."
python /Users/henrikheiland/Desktop/MERFISH/EcoFoundation/R_Functions/bash_Script/run_c2l_script.py \
  --adata_path "$AD_FILE" \
  --infer_csv "$INF_CSV" \
  --output "$OUTPUT_CSV" \
  --maxepochs "$MAXEPOCHS"

# ----------------------------
# Deactivate conda
# ----------------------------
conda deactivate
echo "✅ Done. Output saved to: $OUTPUT_CSV"
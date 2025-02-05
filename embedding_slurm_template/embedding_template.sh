#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=UNLIMITED
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu2
#SBATCH --account=gpu2
#SBATCH --qos gpu
#SBATCH --mem 50GB

# Tells the node to activate your conda env before running job
source activate esm-umap

# Define path variables do not change unless you want to run your own models
EXTRACT_SCRIPT="/mnt/VEIL/tools/embeddings/models/extract.py"
MODEL_PATH="/mnt/VEIL/tools/embeddings/models/esm2_t36_3B_UR50D.pt"

#Path to input amino acid fasta file and output directory. List full path
INPUT_FILE="/work/user/fasta_file_location.fa"
OUTPUT_DIR="/work/user/output/dir"

#Run command for the embeddings to generate mean-representation of the protein
python $EXTRACT_SCRIPT $MODEL_PATH $INPUT_FILE $OUTPUT_DIR --repr_layers 36 --include mean

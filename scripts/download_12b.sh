#!/bin/bash
#SBATCH -t 100
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1

#Loading modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

pip install transformers torch tqdm numpy datasets accelerate matplotlib scikit-learn

python $HOME/MIA-metric-thesisproject/utils/download_12b.py

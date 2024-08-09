#!/bin/bash
#SBATCH -t 20
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gpus-per-node=4

#Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

#Install pip modules
pip install transformers torch tqdm numpy datasets accelerate matplotlib scikit-learn

#Run script
python $HOME/MIA-metric-thesisproject/run.py --model "EleutherAI/pythia-12b" --dataset "swj0419/WikiMIA" --split_name "WikiMIA_length128"

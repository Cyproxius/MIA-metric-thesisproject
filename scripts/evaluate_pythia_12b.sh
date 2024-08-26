#!/bin/bash
#SBATCH -t 360
#SBATCH -N 2
#SBATCH -p gpu
#SBATCH --gpus-per-node=4

#Loading modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

#Install pip modules
pip install transformers torch tqdm numpy datasets accelerate matplotlib scikit-learn

#Run script
accelerate launch --config_file $HOME/.cache/huggingface/accelerate/default_config.yaml $HOME/MIA-metric-thesisproject/run.py --model "EleutherAI/pythia-12b" --dataset "swj0419/WikiMIA" --split_name "WikiMIA_length128"

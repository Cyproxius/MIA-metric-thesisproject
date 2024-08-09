# Imports
import logging
logging.basicConfig(level='ERROR')
from pathlib import Path
import os
from eval import *
from experiment_utils import *
from model_utils import *
from unlearning import *
import argparse

def get_experiment_args(args):

    model = args.model
    model_dir = args.model_dir
    dataset_name = args.dataset
    split_name = args.split_name
    threshold = args.threshold

    output_dir = f"/gpfs/home3/mkoopmans/experiment_output/{model}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    experiment_args = ExperimentArgs(
            model = model,
            output_dir = output_dir,
            # Directory in which to store models locally (to prevent having to download for each experiment)
            model_dir_prefix = model_dir,
            data = dataset_name,
            split_name = split_name,
            threshold = threshold
        )
    # swj0419/WikiMIA
    # Cyproxius/GutenbergMIA_temporal
    # GutenbergMIA
    # NIH_ExPorterMIA_temporal
    # NIH_ExPorterMIA

    return experiment_args

    
def perform_experiment(parse_args, experiment_args):
    lrs = parse_args.learning_rates
    ul_steps = parse_args.unlearning_steps
    batch_sizes = parse_args.batch_sizes
    
    unlearning_args = UnlearningArgs(
            lr=lrs[0],
            steps = ul_steps[0],
            batch_size = batch_sizes[0],
            include_learning = parse_args.include_learning,
            metric = parse_args.metric,
            num_repeats = parse_args.num_repeats
    )
    
    experiment = Experiment(experiment_args, unlearning_args)
    
    # Check if we need to perform a grid search
    if len(lrs)*len(ul_steps)*len(batch_sizes) > 1:
        experiment.run_gridsearch(lrs, ul_steps, batch_sizes)
    else:
        experiment.run_experiment(unlearning_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A description of what your program does.")
    
    # Experiment arguments
    parser.add_argument('--model', '-m', type=str, required=True, help='Model name as presented on HuggingFace. Example: EleutherAI/pythia-2.8b')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='Dataset to perform MIA on, as presented on HuggingFace. Example: swj0419/WikiMIA')
    parser.add_argument('--split_name', '-s', type=str, required=True, help='Split name, as presented on HuggingFace. For these experiments this is the length split. Example: WikiMIA_length128')
    parser.add_argument('--model_dir', '-md', type=str, default="/gpfs/home3/mkoopmans/base_models_dupe/", help='Directory to save the downloaded models to.')
    parser.add_argument('--threshold', '-t', type=int, default=160, help='Threshold value for cutting off tokenizer (default: 160)')

    # Unlearning arguments
    parser.add_argument('--learning_rates', '-lrs', type=list, default=[1e-6], help='List of all learning rates to explore in grid search. Default [1e-6]')
    parser.add_argument('--unlearning_steps', '-ul', type=list, default=[4], help='List of all unlearning steps to explore in grid search. Default [4]')
    parser.add_argument('--batch_sizes', '-bs', type=list, default=[16], help='List of all batch sizes to explore in grid search. Default [16]')
    parser.add_argument('--include_learning', '-ic', type=bool, default=False, help='Whether to use a trained model as a reference instead of the base model. Default: False')
    parser.add_argument('--metric', '-mt', type=str, default="All", help='Metric to use in calculating MIA. Choose from ["PPL", "Min_K", "Min_K++", "All"]. Default: "All"')
    parser.add_argument('--num_repeats', '-n', type=str, default="1", help='Number of times to repeat each experiment. Defaults to 1.')
    
    # Parsing arguments
    args = parser.parse_args()
    
    exp_args = get_experiment_args(args)
    perform_experiment(args, exp_args)

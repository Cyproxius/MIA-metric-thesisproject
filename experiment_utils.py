from model_utils import *
from data_utils import *
from unlearning import *
from collections import defaultdict
from hyperopt import hp
from eval import *
import numpy as np
import json
import math
import copy
import os
import re

class Experiment:

  def __init__(self, experiment_args, unlearning_args):

    self.experiment_args = experiment_args
    self.unlearning_args = unlearning_args

  def experiment_loop(self):
    split = f"{self.experiment_args.data_name}_length{self.experiment_args.length}"

    base_model, tokenizer = load_base_model(self.experiment_args.model_dir_prefix, self.experiment_args.model)
    dataloader = load_unlearn_dataset(self.experiment_args, tokenizer, self.unlearning_args.batch_size, split=split)

    all_MIM_scores = []
    all_labels = []

    UL_PPL_vals = []
    ref_PPL_vals = []

    UL_min_K_vals = []
    ref_min_K_vals = []

    UL_min_K_plusplus_vals = []
    ref_min_K_plusplus_vals = []

    for i, (batch_inputs, batch_labels) in tqdm(enumerate(dataloader)):
      all_labels += batch_labels

      unlearned_model = copy.deepcopy(base_model)
      # Unlearn data and calculate PPL values
      for _ in range(self.unlearning_args.steps):
        unlearned_model = unlearn_dataslice(unlearned_model, tokenizer, batch_inputs, self.unlearning_args)

      UL_PPL_vals += calculate_PPL_values(unlearned_model, tokenizer, batch_inputs)
      UL_min_K_vals += calculate_min_K_scores(unlearned_model, tokenizer, batch_inputs)
      UL_min_K_plusplus_vals += calculate_min_K_plusplus_scores(unlearned_model, tokenizer, batch_inputs)

      # Delete unlearned model to reduce memory usage
      del unlearned_model

      if self.unlearning_args.include_learning:
        learned_model = copy.deepcopy(base_model)
        for _ in range(self.unlearning_args.steps):
          learned_model = learn_dataslice(learned_model, tokenizer, batch_inputs, self.unlearning_args)

        ref_PPL_vals += calculate_PPL_values(learned_model, tokenizer, batch_inputs)
        ref_min_K_vals += calculate_min_K_scores(learned_model, tokenizer, batch_inputs)
        ref_min_K_plusplus_vals += calculate_min_K_plusplus_scores(learned_model, tokenizer, batch_inputs)
        del learned_model

      else:
        ref_PPL_vals += calculate_PPL_values(base_model, tokenizer, batch_inputs)
        ref_min_K_vals += calculate_min_K_scores(base_model, tokenizer, batch_inputs)
        ref_min_K_plusplus_vals += calculate_min_K_plusplus_scores(base_model, tokenizer, batch_inputs)

    if self.unlearning_args.metric == 'PPL':
      all_MIM_scores = calculate_MIM_scores(ref_PPL_vals, UL_PPL_vals)
    elif self.unlearning_args.metric == 'Min_K':
      all_MIM_scores = calculate_MIM_scores(ref_min_K_vals, UL_min_K_vals)
    elif self.unlearning_args.metric == 'Min_K++':
      all_MIM_scores = calculate_MIM_scores(ref_min_K_plusplus_vals, UL_min_K_plusplus_vals)
    elif self.unlearning_args.metric == 'All':
      all_MIM_scores = calculate_MIM_scores_combined(ref_PPL_vals, UL_PPL_vals, ref_min_K_vals, UL_min_K_vals, ref_min_K_plusplus_vals, UL_min_K_plusplus_vals)
    
    # Delete models from memory to reduce memory usage
    del base_model

    # Check if any MIM scores are inf, then the model has been lobotomized. In that case, save the results as NaN
    if any([math.isinf(num) for d in all_MIM_scores for key, num in d.items()]):
      results_dict = self.get_results_dict_nan(all_MIM_scores)
    else:
      results_dict = self.get_results_dict(all_MIM_scores, all_labels)

    return results_dict
  
  def get_results_dict_nan(self, MIM_scores):
    print(f"Encountered inf values in MIM calculation")
    results_dict = {metric: float('nan') for metric in MIM_scores[0].keys()}
    results_dict["params"] = {
      "lr": copy.deepcopy(self.unlearning_args.lr),
      "steps": copy.deepcopy(self.unlearning_args.steps),
      "batch_size": copy.deepcopy(self.unlearning_args.batch_size)
    } 
    return results_dict
  
  def get_results_dict(self, MIM_scores, all_labels):
    results_dict = {metric: sweep(np.array([i[metric] for i in MIM_scores]), np.array(all_labels)) for metric in MIM_scores[0].keys()}
    results_dict["params"] = {
      "lr": copy.deepcopy(self.unlearning_args.lr),
      "steps": copy.deepcopy(self.unlearning_args.steps),
      "batch_size": copy.deepcopy(self.unlearning_args.batch_size)
    } 
    return results_dict

  def run_experiment(self, unlearning_args):
    self.unlearning_args = unlearning_args
    results_dict = self.experiment_loop()

    self.save_experiment_data([results_dict], "single")

  def run_gridsearch(self, lrs, steps, batches):
    results_list = []
    n = self.unlearning_args.num_repeats

    for lr in lrs:
      self.unlearning_args.lr = lr
      for step in steps:
        self.unlearning_args.steps = step
        for batch in batches:
          self.unlearning_args.batch_size = batch

          temp_dict = defaultdict(list)
          for j in range(n):
            print(f'Running epxeriment {j+1} out of {n}')
            result_dict = self.experiment_loop()
            metric_names = [i for i in result_dict.keys() if i != "params"]

            for metric in metric_names:
              temp_dict[metric].append(result_dict[metric])
          
          processed_result_dict = {metric: {'mean': np.mean(values), 'std': np.std(values)} for (metric, values) in temp_dict.items()}
          processed_result_dict["params"] = result_dict["params"]
          results_list.append(processed_result_dict)


    print(f"Experiment data: {results_list}")

    self.save_experiment_data(results_list, "grid_search")


  def save_experiment_data(self, data, experiment_type):
    output_dir = self.experiment_args.output_dir
    if experiment_type == "single":
      output_file = f"{experiment_type}_lr{data[0]['params']['lr']}_steps{data[0]['params']['steps']}_batchsize{data[0]['params']['batch_size']}.json"
    else:
      last_index = self.get_last_experiment_index(output_dir, f"{self.experiment_args.data_name}_length{self.experiment_args.length}_{experiment_type}_", ".json")
      output_file = f"{self.experiment_args.data_name}_length{self.experiment_args.length}_{experiment_type}_{last_index+1}.json"

    # Save data_list to a JSON file
    with open(f'{output_dir}/{output_file}', 'w') as f:
        json.dump(data, f, indent=4)

  def get_last_experiment_index(self, directory, prefix, suffix):

    files = os.listdir(directory)
    # Use regex to find all filenames that match the pattern f"{prefix}{i}{suffix}"
    pattern = re.compile(rf"{prefix}(\d+){suffix}")
    # Extract the numerical part of the filenames
    indices = [int(pattern.match(file).group(1)) for file in files if pattern.match(file)]
    # Return the maximum index, or 0 if no matching files are found
    return max(indices) if indices else 0


class ExperimentArgs:
  def __init__(self, model, output_dir, model_dir_prefix, input_name, label_name, data, data_name, length):
      self.model = model
      self.output_dir = output_dir
      self.model_dir_prefix = model_dir_prefix
      self.input_name = input_name
      self.label_name = label_name
      self.data = data
      self.data_name = data_name
      self.length = length

class UnlearningArgs:
  def __init__(self, lr, steps, batch_size, include_learning, metric, num_repeats):
    self.lr = lr
    self.steps = steps
    self.batch_size = batch_size
    self.include_learning = include_learning
    self.metric = metric
    self.num_repeats = num_repeats
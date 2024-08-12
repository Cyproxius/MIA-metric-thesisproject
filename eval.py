import logging
logging.basicConfig(level='ERROR')
import numpy as np
from tqdm import tqdm
import torch
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, roc_auc_score
import matplotlib
import random

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def calculatePerplexity(sentence, model, tokenizer, accelerator):
    """
    exp(loss)
    """
    input_ids = torch.tensor(sentence).unsqueeze(0)
    print(f"Input id device: {input_ids.device}")
# print(f"Input ids shape: {input_ids.size()}")
    model.eval()
    with torch.no_grad():
        print(f'Model device torch no_grad: {model.device}')
        print(f'Input_ids device torch.no_grad {input_ids.device}')
        outputs = model(input_ids, labels=input_ids)
        print(f'Model device after inference: {model.device}')
    outputs = accelerator.gather_for_metrics(outputs)
    print(outputs)
    loss, logits = outputs[:2]
    print(f"Loss device: {loss.device}")
    print(f'Model device: {model.device}')
    print(f'Logits device: {logits.device}')
    loss, logits = accelerator.prepare(loss, logits)
    print(f'Loss and logits after accelerate: {loss.device}, {logits.device}')
    # print(f"Loss shape: {loss.size()}")
    # print(f"Loss: {loss}")
    # print(f"Logits dimension: {np.array(logits).shape()}")

    # print(f"Loss: {loss}")
    # print(f"Logits: {logits}")
    # print(f"Logits elem: {logits[0]}")
    # print(f"Logits elem size: {logits[0][0].size()}")
    '''
    extract logits:
    '''
    # Apply softmax to the logits to get probabilities
    # probabilities = torch.nn.functional.log_softmax(logits[0][0], dim=-1)

    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    print(f'Probabilities device: {probabilities.device}')
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    print(f'input_ids_processed device: {input_ids_processed.device}')
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)

    # stuff for Min-K%++ calculation 
    probs = torch.nn.functional.softmax(logits[0, :-1], dim=-1)
    print(f'Probs device: {probs.device}')
    log_probs = torch.nn.functional.log_softmax(logits[0, :-1], dim=-1)
    print(f'log_probs device: {log_probs.device}')
    token_log_probs = log_probs.gather(dim=-1, index=input_ids_processed.unsqueeze(-1)).squeeze(-1)
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
    mink_plus = (token_log_probs - mu) / sigma.sqrt()

    return torch.exp(loss).item(), all_prob, mink_plus#, loss.item()

def calculate_min_K_plusplus_scores(model, tokenizer, text_batch):
    min_K_plusplus_values = []
    for text in text_batch:
        min_k_plus = calculatePerplexity(text, model, tokenizer, gpu=model.device)[2]
        # Change ratio here
        k_length = int(len(min_k_plus)*0.3)
        topk_prob = np.sort(min_k_plus.cpu())[:k_length]
        min_K_score = -np.mean(topk_prob).item()
        min_K_plusplus_values.append(min_K_score)

    return min_K_plusplus_values

def calculate_min_K_scores(model, tokenizer, text_batch):
    min_K_values = []
    for text in text_batch:
        all_prob = calculatePerplexity(text, model, tokenizer, gpu=model.device)[1]
        # Change ratio here
        k_length = int(len(all_prob)*0.2)
        topk_prob = np.sort(all_prob)[:k_length]
        min_K_score = -np.mean(topk_prob).item()
        min_K_values.append(min_K_score)

    return min_K_values

def calculate_PPL_values(model, tokenizer, text_batch, accelerator):
    PPL_values = []
    print(f'Model device (calculate_PPL_values): {model.device}')
    print(f'Text batch device: {text_batch.device}')
    for text in text_batch:
        print(f'Text device: {text.device}')
        PPL = calculatePerplexity(text, model, tokenizer, accelerator)[0]
        PPL_values.append(PPL)
    return PPL_values

def calculate_MIM_scores(base_PPLs, unlearned_PPLs):
    scores = []
    for (p_base, p_ul) in zip(base_PPLs, unlearned_PPLs):

        MIM_diff = (p_ul - p_base)
        MIM_plus = (p_ul + p_base)
        MIM_only_ul = p_ul
        MIM_only_base = p_base
        MIM_multiplied = (p_ul - p_base) * p_base
        MIM_mult_UL = (p_ul - p_base) * p_ul
        MIM_squared_ind = (p_ul**2) - (p_base**2)

        scores.append({"Difference": MIM_diff, "Addition":MIM_plus, "Base model only":MIM_only_base, "Unlearning only": MIM_only_ul, "Multiplied base":MIM_multiplied, "Multiplied UL": MIM_mult_UL, "Squared difference": MIM_squared_ind})

    return scores


def calculate_MIM_scores_combined(ref_PPLs, UL_PPLs, ref_Ks, UL_Ks, ref_Kplusplus, UL_Kpluslus):
    scores = []
    for (p_base, p_ul, k_base, k_ul, k_plusplus_base, k_plusplus_ul) in zip(ref_PPLs, UL_PPLs, ref_Ks, UL_Ks, ref_Kplusplus, UL_Kpluslus):

        MIM_PPL_diff = (p_ul - p_base)
        MIM_PPL_base = p_base

        MIM_Min_K_diff = (k_base - k_ul)
        MIM_Min_K_base = k_base
        
        MIM_Min_Kplusplus_diff = (k_plusplus_base - k_plusplus_ul)
        MIM_Min_Kplusplus_base = k_plusplus_base
        
        scores.append({"PPL difference": MIM_PPL_diff, "PPL base": MIM_PPL_base, "Min-K difference": MIM_Min_K_diff, "Min-K base": MIM_Min_K_base, "Min-K++ difference": MIM_Min_Kplusplus_diff, "Min-K++ base": MIM_Min_Kplusplus_base})
    
    return scores

# plot data 
def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    auc_score = roc_auc_score(x, -score)
    # acc = np.max(1-(fpr+(1-tpr))/2)
    return auc_score
    # return fpr, tpr, auc_score, acc


def do_plot(prediction, answers, sweep_fn=sweep, metric='auc', legend="", output_dir=None):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """
    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr<.05)[0][-1]]
    # bp()
    print('Attack %s   AUC %.4f, Accuracy %.4f, TPR@5%%FPR of %.4f\n'%(legend, auc,acc, low))

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f'%auc
    elif metric == 'acc':
        metric_text = 'acc=%.3f'%acc

    plt.plot(fpr, tpr, label=legend+metric_text)
    return legend, auc,acc, low


def fig_fpr_tpr(all_output, output_dir):
    print("output_dir", output_dir)
    answers = []
    metric2predictions = defaultdict(list)
    for ex in all_output:
        answers.append(ex["label"])
        for metric in ex["pred"].keys():
            if ("raw" in metric) and ("clf" not in metric):
                continue
            metric2predictions[metric].append(ex["pred"][metric])
    
    plt.figure(figsize=(4,3))
    with open(f"{output_dir}/auc.txt", "w") as f:
        for metric, predictions in metric2predictions.items():
            print(f"Metric: {metric}\n Predictions: {predictions}")
            legend, auc, acc, low = do_plot(predictions, answers, legend=metric, metric='auc', output_dir=output_dir)
            f.write('%s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f\n'%(legend, auc, acc, low))

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5,1)
    plt.ylim(1e-5,1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)
    plt.savefig(f"{output_dir}/auc.png")


def load_jsonl(input_path):
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in tqdm(f)]
    random.seed(0)
    random.shuffle(data)
    return data

def dump_jsonl(data, path):
    with open(path, 'w') as f:
        for line in tqdm(data):
            f.write(json.dumps(line) + "\n")

def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in tqdm(f)]

def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data

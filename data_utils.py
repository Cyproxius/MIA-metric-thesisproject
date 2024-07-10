import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from eval import *

class DatasetFromDict(Dataset):
    def __init__(self, data_list, input_name, label_name):
        """
        Args:
            data_list (list): List of dictionaries with keys 'input' and 'label'
        """
        self.data_list = data_list
        self.input_name = input_name
        self.label_name = label_name

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_item = self.data_list[idx]
        input_data = data_item[self.input_name]
        label = data_item[self.label_name]
        return torch.tensor(input_data), torch.tensor(label)

def tokenize_datalist(datalist, tokenizer, args, threshold=0):
    # Tokenize all data inputs 
    tokenized_datalist = [
        {args.input_name: tokenizer.encode(item[args.input_name]),
         args.label_name: item[args.label_name]}
            for item in datalist
    ]

    # Cutoff data (just for WikiMIA which isn't sorted on token length)
    tokenized_datalist_cutoff = [
        {args.input_name: item[args.input_name][:threshold],
         args.label_name: item[args.label_name]}
            for item in tokenized_datalist if len(item[args.input_name]) >= threshold
    ]

    return tokenized_datalist_cutoff

def load_unlearn_dataset(args, tokenizer, batch_size, split):
    
    # load data from huggingface
    huggingface_dataset = load_dataset(args.data, split=split)

    datalist = convert_huggingface_data_to_list_dic(huggingface_dataset)

    # TODO for normal data, set threshold to 0
    tokenized_datalist = tokenize_datalist(datalist, tokenizer, args, threshold=160)

    dataset = DatasetFromDict(tokenized_datalist, args.input_name, args.label_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
import torch
from datasets import load_dataset
import itertools
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from eval import *
import gc
import sys

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

def tokenize_datalist(datalist, tokenizer, args, threshold=0, genres=None):
    # Tokenize all data inputs
    if genres:
        tokenized_datalist = [
            {"input": tokenizer.encode(item["input"]),
            "label": item["label"]}
                for item in datalist if (item['subject'] == genres[0] and item['label'] == 1) or (item['subject'] == genres[1] and item['label'] == 0)
        ]
    else:    
        tokenized_datalist = [
            {"input": tokenizer.encode(item["input"]),
            "label": item["label"]}
                for item in datalist
        ]
    if threshold:
        # Cutoff dataset 
        tokenized_datalist_cutoff = [
            {"input": item["input"][:threshold],
            "label": item["label"]}
                for item in tokenized_datalist if len(item["input"]) >= threshold
        ]
        return tokenized_datalist_cutoff
    # else:
    #     return tokenized_datalist
    else:
        max_length = max([len(d["input"]) for d in tokenized_datalist])
        print(f"Max length: {max_length}")
        padded_datalist = [
            {"input": F.pad(torch.tensor(item["input"]), (0, max_length - len(item["input"]))),
             "label": item["label"]}
                for item in tokenized_datalist
        ]
        return padded_datalist

def load_unlearn_dataset(args, tokenizer, batch_size, split):
    
    # load data from huggingface
    huggingface_dataset = load_dataset(args.data, split=split)

    datalist = convert_huggingface_data_to_list_dic(huggingface_dataset)

    # TODO for normal data, set threshold to 0
    tokenized_datalist = tokenize_datalist(datalist, tokenizer, args, threshold=args.threshold)

    dataset = DatasetFromDict(tokenized_datalist, "input", "label")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def load_gutenbergmia_genres(args, tokenizer, batch_size):
    split = args.split_name
    genres = ["Adventure stories", "Love stories", "Historical fiction", "Western stories"]
    genre_dataloader_dict = {}
    genre_combinations = itertools.product(genres, repeat=2)

    full_dataset = load_dataset(args.data, split=split)
    datalist = convert_huggingface_data_to_list_dic(full_dataset)
    
    print("Creating dataloaders for all permutations")
    for genre_combination in genre_combinations:
        print("Creating new dataloader")
        tokenized_dl_comb = tokenize_datalist(datalist, tokenizer, args, threshold=args.threshold, genres=genre_combination)
        dataset_comb = DatasetFromDict(tokenized_dl_comb, "input", "label")
        dataloader_comb = DataLoader(dataset_comb, batch_size=batch_size, shuffle=True)

        genre_dataloader_dict[genre_combination] = dataloader_comb
    print(f"Finished creating {len(genre_dataloader_dict.keys())} dataloaders")
    return genre_dataloader_dict

def print_torch_memory():
    if torch.cuda.is_available():
        print("PyTorch GPU Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0)/1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0)/1024**3, 1), "GB")
    else:
        print("CUDA is not available.")

def get_variable_memory_usage():
    # Garbage collection
    gc.collect()
    
    # Get all objects in memory
    all_objects = gc.get_objects()
    
    # Track memory usage
    memory_usage = {}
    for obj in all_objects:
        try:
            # Check if the object is a PyTorch tensor
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                obj_size = obj.element_size() * obj.nelement()
                memory_usage[str(type(obj))] = memory_usage.get(str(type(obj)), 0) + obj_size
            else:
                # Get size of regular Python objects
                obj_size = sys.getsizeof(obj)
                memory_usage[str(type(obj))] = memory_usage.get(str(type(obj)), 0) + obj_size
        except:
            pass

    sorted_memory_usage = sorted(memory_usage.items(), key=lambda x: x[1], reverse=True)
    # Print the memory usage
    for obj_type, size in sorted_memory_usage[:3]:
        print(f"{obj_type}: {round(size/1024**2, 2)} MB")
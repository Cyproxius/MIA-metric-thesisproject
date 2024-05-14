from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
from model_utils import *
from pathlib import Path
from eval import *
import torch
import os 

def load_unlearned_model(experiment_params, unlearning_params):
    model_name = experiment_params.unlearn_model
    print(f"Model name: {model_name}")
    model_dir = "/dbfs/mnt/ds-data-apps/maris/unlearned_models/" + model_name + f"/lr_{unlearning_params.lr}_epochs_{unlearning_params.epochs}"

    if not os.path.exists(model_dir):
        print("Creating unlearned model...")
        model, tokenizer = create_unlearned_model(model_name, model_dir, experiment_params, unlearning_params)
    else:
        print(f"Loading model {model_name} from disk")
        # # Initialize with empty weights to reduce memory usage
        # with init_empty_weights():
        #     config = AutoConfig.from_pretrained(name)
        #     model_empty = AutoModel.from_config(config)

        # model = load_checkpoint_and_dispatch(
        #     model_empty, checkpoint=model_dir, device_map="auto"
        # )
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def create_unlearned_model(model_name, model_dir, experiment_params, unlearning_params):
    
    model, tokenizer = load_base_model(model_name)
    dataset = load_unlearn_dataset(
                                    experiment_params.data,
                                    split=f"WikiMIA_length{experiment_params.length}",
                                    only_members=unlearning_params.only_members
                                   )
    unlearned_model, losses = unlearn_dataset(model, tokenizer, dataset, unlearning_params)
    print(f"Losses: {losses}")

    if unlearning_params.save_model:
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        unlearned_model.save_pretrained(model_dir)
        print("Saved model to disk")

    # if experiment_params.save_loss:
    #     graph_dir = experiment_params.output_dir + "/graphs"
    #     Path(graph_dir).mkdir(parents=True, exist_ok=True)
    #     plt.plot(range(len(losses)), losses)
    #     plt.savefig(graph_dir + f"{model_name}_onlymembers_{'yes' if only_members else 'no'}_lr{params['lr']}_epochs{params['epochs']}.png")

    return unlearned_model, tokenizer

def unlearn_dataset(model, tokenizer, dataset, unlearning_params):
    learning_rate = unlearning_params.lr
    epochs = unlearning_params.epochs
    col_name = unlearning_params.col_name
    device = unlearning_params.device
    model.to(device)
    
    print(f"Putting input data on device {device}")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    print(f"Started unlearning process")
    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0
        print(f"Training epoch {epoch+1}")
        for data in tqdm(dataset):
            optimizer.zero_grad()
            sentence = data[col_name]
            input_data = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0).to(device)  # Adding batch dimension
            output = model(input_data)
            loss = -criterion(output.logits.squeeze(0), input_data.squeeze(0).long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_losses.append(total_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss}")

    return model, epoch_losses

def load_unlearn_dataset(dataset_name, split, only_members=True):
  
    # TODO filter so that only member data is kept in unlearning dataset
    if "jsonl" in dataset_name:
        data = load_jsonl(f"{dataset_name}")
    else: # load data from huggingface
        dataset = load_dataset(dataset_name, split=split)
        data = convert_huggingface_data_to_list_dic(dataset)
        if only_members:
            data = [i for i in data if i["label"] == 1]
        # print(data[:3])
    return data

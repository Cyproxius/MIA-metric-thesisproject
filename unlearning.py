from data_utils import *
from torch.utils.checkpoint import checkpoint
import torch

def learn_dataslice(model, tokenizer, sentences, args):
    learning_rate = args.lr
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    optimizer.zero_grad()
    input_data = torch.tensor(sentences).to(device)  # Adding batch dimension
    output = model(input_data)
    # loss = -criterion(output.logits.squeeze(0), input_data.squeeze(0).long())
    # Add a minus do to gradient ascent instead of descent
    loss = output[0]
    loss.mean().backward()
    torch.cuda.empty_cache()
    optimizer.step()
    
    del optimizer
    torch.cuda.empty_cache()
    return model

def unlearn_dataslice(model, tokenizer, sentences, args):
    learning_rate = args.lr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    optimizer.zero_grad()
    # print_torch_memory()
    input_data = sentences.clone().detach().to(device)  # Adding batch dimension
    
    # print("Memory usage before forward pass")
    # print_torch_memory()
    # torch.cuda.empty_cache()
    # print("Memory usage after forward pass")
    # print_torch_memory()

    output = model(input_data)
    # output = checkpoint(model, input_data)

    # Add a minus do to gradient ascent instead of descent
    loss = -output[0]
    loss.mean().backward()
    torch.cuda.empty_cache()
    # print("Memory usage before calling optimizer.step()")
    # print_torch_memory()
    optimizer.step()
    # print("Memory usage after calling optimizer.step()")
    # print_torch_memory()

    del optimizer
    torch.cuda.empty_cache()
    return model
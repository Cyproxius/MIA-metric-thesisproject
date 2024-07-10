from data_utils import *
import torch

def learn_dataslice(model, tokenizer, sentences, args):
    learning_rate = args.lr
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
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
    optimizer.step()
            
    return model

def unlearn_dataslice(model, tokenizer, sentences, args):
    learning_rate = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    optimizer.zero_grad()
    input_data = torch.tensor(sentences).to(device)  # Adding batch dimension
    output = model(input_data)
    # loss = -criterion(output.logits.squeeze(0), input_data.squeeze(0).long())
    # Add a minus do to gradient ascent instead of descent
    loss = -output[0]
    # if args.include_learning:
    #     loss = -loss
    loss.mean().backward()
    optimizer.step()
            
    return model
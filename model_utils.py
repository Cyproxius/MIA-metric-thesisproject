from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel, PreTrainedTokenizerFast
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import torch
import os 

# Helper functions to load models
def load_base_model(name):
    model_dir = "/dbfs/mnt/ds-data-apps/maris/base_models_dupe/" + name

    if not os.path.exists(model_dir):
      print(f"Downloading model {name}")
      download_model(model_dir, name)

    model = AutoModelForCausalLM.from_pretrained(model_dir, return_dict=True, device_map='auto')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name)

    return model, tokenizer

def download_model(model_dir, name):

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(name, return_dict=True, device_map='auto')
    model.save_pretrained(model_dir)

    print(f"Model {name} downloaded and saved to {model_dir}")

def generate_text(llm_model, tokenizer, input_text, num_words=20, temperature=1.0):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  llm_model = llm_model.to(device)
  # tokenizer = tokenizer.to(device)
  # Set the model to evaluation mode
  llm_model.eval()
  
  # Tokenize input text
  tokenized_input = tokenizer.encode(input_text, return_tensors="pt")
  
  # Generate additional words
  with torch.no_grad():
      # for _ in range(num_words):
        # Generate output tokens
        output = llm_model.generate(
            tokenized_input.to(device),
            do_sample=True,
            temperature=temperature,
            max_length=len(tokenized_input[0]) + 50,  # Maximum output length
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
        )
        
        # Decode generated tokens into text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Update input text for next iteration
        input_text += " " + generated_text
        
        # Update tokenized input for next iteration
        tokenized_input = tokenizer.encode(input_text, return_tensors="pt")

  return input_text

def calculatePerplexity(sentence, model, tokenizer, gpu):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    # print(f"Input ids: {input_ids}")
    input_ids = input_ids.to(gpu)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    # print(f"Loss: {loss}")
    # print(f"Logits: {logits}")
    
    '''
    extract logits:
    '''
    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    # print(f"Probabilities: {probabilities}")
    # probabilities = torch.nn.functional.softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    
    # print(f"PPL: {torch.exp(loss).item()}")
    # print(f"Probability: {loss.item()}")
    return torch.exp(loss).item(), all_prob, loss.item()

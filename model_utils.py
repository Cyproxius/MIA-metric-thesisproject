from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXConfig, GPTNeoXForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from pathlib import Path
import torch
import os 

# Helper functions to load models
def load_base_model(model_dir_prefix, name):
  
  #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model_dir = model_dir_prefix + name

  if not os.path.exists(model_dir):
    print(f"Downloading model {name}")
    download_model(model_dir, name)

  # Code to load models using accelerate, gives error
  # config = GPTNeoXConfig.from_pretrained(name)
  # with init_empty_weights():
  #   model = GPTNeoXForCausalLM(config)
  # model = load_checkpoint_and_dispatch(
  #   model, checkpoint=model_dir
  # )
  model = AutoModelForCausalLM.from_pretrained(model_dir, return_dict=True)
  # Halving the models precision, diminishes results
  # model = model.half()
  model.eval()
  tokenizer = AutoTokenizer.from_pretrained(name)

  return model, tokenizer

def download_model(model_dir, name):

  Path(model_dir).mkdir(parents=True, exist_ok=True)
  model = AutoModelForCausalLM.from_pretrained(name, return_dict=True, device_map='auto')
  model.save_pretrained(model_dir)
  del model
  print(f"Model {name} downloaded and saved to {model_dir}")

def generate_text(llm_model, tokenizer, input_text, num_words=20, temperature=1.0):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  llm_model = llm_model.to(device)
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

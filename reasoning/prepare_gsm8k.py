"""
Downloads and tokenizes GSM8K dataset for Chain-of-Thought fine-tuning.
Run as: python prepare_gsm8k.py
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# setup
script_dir = os.path.dirname(__file__)
local_dir = os.path.join(script_dir, "data")
os.makedirs(local_dir, exist_ok=True)

# tokenizer
enc = tiktoken.get_encoding('gpt2')
eot = enc._special_tokens['<|endoftext|>']

def format_example(example):
    """Format GSM8K example as Chain-of-Thought"""
    question = example['question'].strip()
    answer = example['answer'].strip()
    
    # extract reasoning and final answer
    if '####' in answer:
        reasoning = answer.split('####')[0].strip()
        final_answer = answer.split('####')[1].strip()
    else:
        reasoning = answer
        final_answer = "Unknown"
    
    # format as CoT
    formatted = f"Question: {question}\n\nLet me solve this step by step:\n{reasoning}\n\nTherefore, the answer is {final_answer}."
    return formatted

def tokenize_and_save(examples, split_name):
    """Tokenize examples and save to file"""
    all_tokens = []
    
    print(f"Processing {len(examples)} {split_name} examples...")
    for example in tqdm(examples):
        formatted_text = format_example(example)
        tokens = [eot] + enc.encode_ordinary(formatted_text) + [eot]
        all_tokens.extend(tokens)
    
    # save as numpy array
    tokens_np = np.array(all_tokens, dtype=np.uint16)
    filepath = os.path.join(local_dir, f'gsm8k_{split_name}.npy')
    np.save(filepath, tokens_np)
    
    print(f"Saved {len(all_tokens)} tokens to {filepath}")
    return len(all_tokens)

def main():
    """Download and prepare GSM8K dataset"""
    print("Downloading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")
    
    train_examples = list(dataset['train'])
    test_examples = list(dataset['test'])
    
    print(f"Found {len(train_examples)} train, {len(test_examples)} test examples")
    
    # show example
    sample = format_example(train_examples[0])
    print(f"\nExample formatting:\n{sample[:200]}...\n")
    
    # process and save
    train_tokens = tokenize_and_save(train_examples, 'train')
    test_tokens = tokenize_and_save(test_examples, 'test')
    
    print(f"\nDataset ready! Train: {train_tokens:,} tokens, Test: {test_tokens:,} tokens")

if __name__ == "__main__":
    main()
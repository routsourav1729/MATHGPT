"""
Simple dataloader for GSM8K Chain-of-Thought fine-tuning.
Compatible with original src/dataloader.py interface.
"""

import os
import numpy as np
import torch
import tiktoken

script_dir = os.path.dirname(__file__)

class DataLoaderLite:
    """Simple dataloader for GSM8K reasoning data"""

    def __init__(self, B, T, process_rank, num_processes, split='train'):
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'test'}
        
        # load tokenized data
        data_root = os.path.join(script_dir, "data")
        filepath = os.path.join(data_root, f'gsm8k_{split}.npy')
        self.tokens = torch.tensor(np.load(filepath).astype(np.int32), dtype=torch.long)
        
        # get end-of-text token for splitting examples
        enc = tiktoken.get_encoding('gpt2')
        self.eot = enc._special_tokens['<|endoftext|>']
        
        # split into individual examples
        self._split_examples()
        
        master_process = process_rank == 0
        if master_process:
            print(f'found {len(self.examples)} examples for split {split}')
        self.reset()

    def _split_examples(self):
        """Split tokens into individual reasoning examples"""
        examples = []
        current_example = []
        
        for token in self.tokens:
            if token == self.eot and len(current_example) > 0:
                if len(current_example) <= self.T:  # filter by max length
                    examples.append(torch.tensor(current_example, dtype=torch.long))
                current_example = []
            else:
                current_example.append(token.item())
        
        # add final example
        if len(current_example) > 0 and len(current_example) <= self.T:
            examples.append(torch.tensor(current_example, dtype=torch.long))
        
        self.examples = examples

    def reset(self):
        """Reset to beginning and optionally shuffle"""
        self.current_idx = 0
        if hasattr(self, 'examples'):
            # shuffle examples for training
            if self.num_processes == 1:  # simple shuffle for single process
                indices = torch.randperm(len(self.examples))
                self.examples = [self.examples[i] for i in indices]

    def next_batch(self):
        """Get next batch in same format as original dataloader"""
        batch_examples = []
        
        # collect B examples
        for _ in range(self.B):
            if self.current_idx >= len(self.examples):
                self.reset()
            
            example = self.examples[self.current_idx]
            batch_examples.append(example)
            self.current_idx += 1
        
        # pad to same length T
        x_batch = torch.full((self.B, self.T), self.eot, dtype=torch.long)
        y_batch = torch.full((self.B, self.T), self.eot, dtype=torch.long)
        
        for i, example in enumerate(batch_examples):
            # input: all tokens except last
            # target: all tokens except first (shifted by 1)
            seq_len = min(len(example) - 1, self.T)
            if seq_len > 0:
                x_batch[i, :seq_len] = example[:seq_len]
                y_batch[i, :seq_len] = example[1:seq_len+1]
        
        return x_batch, y_batch
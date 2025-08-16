"""
Generate Chain-of-Thought reasoning for math problems using fine-tuned model.
Run as: python inference.py --question "Your math problem here"
"""

import sys
import torch
import torch.nn.functional as F
import tiktoken
import argparse

# import model from src
sys.path.append('../src')
from model import GPT

class GSM8KInference:
    """Generate step-by-step reasoning for math problems"""

    def __init__(self, model, token_encoder, device):
        self.model = model
        self.token_encoder = token_encoder
        self.device = device
        self.device_type = 'cuda' if device.startswith('cuda') else 'cpu'

    def solve_problem(self, question, max_tokens=200):
        """Generate step-by-step solution for a math problem"""
        self.model.eval()
        
        # format prompt like training data
        prompt = f"Question: {question}\n\nLet me solve this step by step:\n"
        tokens = self.token_encoder.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # generate solution
        sample_rng = torch.Generator(device=self.device).manual_seed(42)
        
        for _ in range(max_tokens):
            with torch.no_grad():
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, _ = self.model(tokens)
                
                # sample next token
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)
                next_token = torch.gather(topk_indices, -1, ix)
                tokens = torch.cat([tokens, next_token], dim=1)
                
                # stop at end of text or reasonable stopping point
                if next_token.item() == self.token_encoder._special_tokens['<|endoftext|>']:
                    break
        
        # decode and return solution
        full_text = self.token_encoder.decode(tokens[0].tolist())
        
        # extract just the generated part
        if prompt in full_text:
            solution = full_text.split(prompt, 1)[1]
        else:
            solution = full_text
        
        return solution.strip()

def load_model(model_path):
    """Load fine-tuned model from checkpoint"""
    print(f"loading model from {model_path}")
    checkpoint = torch.load(model_path, weights_only=False)
    
    model = GPT(config=checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    
    print(f"loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', type=str, required=True, help="Math question to solve")
    parser.add_argument('--model', type=str, default='./logs/gsm8k_model_final.pt', help="Path to fine-tuned model")
    parser.add_argument('--max_tokens', type=int, default=200, help="Maximum tokens to generate")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # setup device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f'using device: {device}')
    
    # load model
    model = load_model(args.model)
    model.to(device)
    
    # setup inference
    token_encoder = tiktoken.get_encoding('gpt2')
    solver = GSM8KInference(model, token_encoder, device)
    
    # solve problem
    print(f"\nQuestion: {args.question}")
    print(f"\nSolution:")
    solution = solver.solve_problem(args.question, args.max_tokens)
    print(solution)
    
    # try to extract numerical answer
    import re
    numbers = re.findall(r'answer is (\d+(?:\.\d+)?)', solution.lower())
    if numbers:
        print(f"\nExtracted Answer: {numbers[-1]}")

if __name__ == '__main__':
    main()
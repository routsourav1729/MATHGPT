"""
Fine-tune pretrained GPT-2 model on GSM8K Chain-of-Thought reasoning.
Loads model from src/ and fine-tunes on mathematical reasoning data.

Run as: python finetune.py
"""

import os
import sys
import time
import torch
import torch.nn as nn

# import model from src
sys.path.append('../src')
from model import GPT
from dataloader import DataLoaderLite

# setup
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f'using device: {device}')

# hyperparameters
batch_size = 8
context_length = 512
learning_rate = 5e-5
num_epochs = 3
eval_freq = 100
log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)

def load_pretrained_model():
    """Load pretrained model from src/logs/"""
    model_path = '../src/logs/model_95364.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pretrained model not found at {model_path}")
    
    print(f"loading pretrained model from {model_path}")
    checkpoint = torch.load(model_path, weights_only=False)
    
    # create model with same config
    model = GPT(config=checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    
    print(f"loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model

def main():
    # load pretrained model
    model = load_pretrained_model()
    model.to(device)
    
    # setup data loaders
    train_loader = DataLoaderLite(B=batch_size, T=context_length, process_rank=0, num_processes=1, split='train')
    test_loader = DataLoaderLite(B=batch_size, T=context_length, process_rank=0, num_processes=1, split='test')
    
    # setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # training loop
    model.train()
    step = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loader.reset()
        
        # estimate steps per epoch
        steps_per_epoch = len(train_loader.examples) // batch_size
        print(f"epoch {epoch+1}/{num_epochs}, ~{steps_per_epoch} steps")
        
        for _ in range(steps_per_epoch):
            t0 = time.time()
            
            # get batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            # forward pass
            with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
                logits, loss = model(x, y)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            dt = time.time() - t0
            
            # logging
            if step % 50 == 0:
                print(f"step {step:4d} | loss: {loss.item():.4f} | dt: {dt*1000:.2f}ms")
            
            # evaluation
            if step % eval_freq == 0:
                model.eval()
                test_loader.reset()
                test_loss = 0.0
                test_steps = 20
                
                with torch.no_grad():
                    for _ in range(test_steps):
                        x_test, y_test = test_loader.next_batch()
                        x_test, y_test = x_test.to(device), y_test.to(device)
                        
                        with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
                            _, loss_test = model(x_test, y_test)
                        test_loss += loss_test.item()
                
                test_loss /= test_steps
                print(f"test loss: {test_loss:.4f}")
                
                model.train()
            
            step += 1
        
        # save checkpoint after each epoch
        checkpoint = {
            'model': model.state_dict(),
            'config': model.config,
            'epoch': epoch + 1,
            'step': step
        }
        checkpoint_path = os.path.join(log_dir, f'gsm8k_model_epoch_{epoch+1}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"saved checkpoint: {checkpoint_path}")
    
    # save final model
    final_path = os.path.join(log_dir, 'gsm8k_model_final.pt')
    torch.save(checkpoint, final_path)
    
    total_time = time.time() - start_time
    print(f"fine-tuning completed in {total_time/60:.1f} minutes")
    print(f"final model saved to: {final_path}")

if __name__ == "__main__":
    main()
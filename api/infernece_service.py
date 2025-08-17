"""
Inference service that wraps the existing Chain-of-Thought reasoning model.
Loads model and provides simple interface for API.
"""

import sys
import os
import re
import torch
import tiktoken
from pathlib import Path

# Add reason directory to path to import existing code
reason_path = Path(__file__).parent.parent / "reason"
sys.path.append(str(reason_path))

# Import existing inference code
try:
    from inference import load_model, GSM8KInference
except ImportError:
    # Fallback if import structure is different
    from finetune_trainer import GPT


class MathSolver:
    """Wrapper for the fine-tuned mathematical reasoning model"""
    
    def __init__(self, model_path: str = None):
        """Initialize the solver with trained model"""
        
        # Set device
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        
        # Default model path
        if model_path is None:
            model_path = str(reason_path / "logs" / "gsm8k_model_final.pt")
        
        self.model_path = model_path
        
        # Load model and tokenizer
        self.model = self._load_model()
        self.tokenizer = tiktoken.get_encoding('gpt2')
        
        print(f"MathSolver initialized on {self.device}")
    
    def _load_model(self):
        """Load the fine-tuned model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        print(f"Loading model from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        # Import GPT model
        src_path = Path(__file__).parent.parent / "src"
        sys.path.append(str(src_path))
        from model import GPT
        
        # Create and load model
        model = GPT(config=checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
    
    def solve(self, question: str, max_tokens: int = 200) -> dict:
        """
        Solve a mathematical problem with step-by-step reasoning.
        
        Args:
            question: Mathematical word problem
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with reasoning, answer, and confidence
        """
        
        # Format prompt like training data
        prompt = f"Question: {question.strip()}\n\nLet me solve this step by step:\n"
        
        # Generate reasoning
        reasoning = self._generate_reasoning(prompt, max_tokens)
        
        # Extract numerical answer
        answer = self._extract_answer(reasoning)
        
        # Simple confidence score (could be improved)
        confidence = 0.95 if answer else 0.7
        
        return {
            "reasoning": reasoning,
            "answer": answer,
            "confidence": confidence,
            "prompt_used": prompt
        }
    
    def _generate_reasoning(self, prompt: str, max_tokens: int) -> str:
        """Generate step-by-step reasoning"""
        
        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Generate with sampling
        with torch.no_grad():
            generated = tokens
            
            for _ in range(max_tokens):
                # Forward pass
                with torch.autocast(device_type=self.device.split(':')[0], dtype=torch.bfloat16):
                    logits, _ = self.model(generated)
                
                # Sample next token
                logits = logits[:, -1, :] / 0.1  # temperature=0.1
                probs = torch.softmax(logits, dim=-1)
                
                # Top-k sampling
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                next_token_idx = torch.multinomial(topk_probs, num_samples=1)
                next_token = torch.gather(topk_indices, -1, next_token_idx)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop conditions
                if next_token.item() == self.tokenizer._special_tokens['<|endoftext|>']:
                    break
                if generated.size(1) > 1000:  # Safety limit
                    break
        
        # Decode and extract reasoning
        full_text = self.tokenizer.decode(generated[0].tolist())
        
        if prompt in full_text:
            reasoning = full_text.split(prompt, 1)[1]
        else:
            reasoning = full_text
        
        return reasoning.strip()
    
    def _extract_answer(self, reasoning: str) -> str:
        """Extract numerical answer from reasoning text"""
        
        patterns = [
            r"Therefore,?\s*the answer is\s*([+-]?\d+(?:\.\d+)?)",
            r"The answer is\s*([+-]?\d+(?:\.\d+)?)",
            r"####\s*([+-]?\d+(?:\.\d+)?)",
            r"=\s*([+-]?\d+(?:\.\d+)?)(?:\s|$|\.)",
            r"([+-]?\d+(?:\.\d+)?)\s*$"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, reasoning, re.IGNORECASE)
            if matches:
                try:
                    return str(float(matches[-1]))
                except ValueError:
                    continue
        
        return None


# Test function
def test_solver():
    """Test the solver locally"""
    try:
        solver = MathSolver()
        
        test_question = "Sarah has 5 apples and buys 3 more. How many apples does she have?"
        result = solver.solve(test_question)
        
        print(f"Question: {test_question}")
        print(f"Reasoning: {result['reasoning'][:200]}...")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_solver()
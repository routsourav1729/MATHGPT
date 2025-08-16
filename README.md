# GPT-2 with Chain-of-Thought Mathematical Reasoning

A complete implementation of GPT-2 language model with an extension for mathematical reasoning capabilities. This project demonstrates both **foundational language modeling** and **specialized reasoning fine-tuning** in a clean, modular architecture.

## 🏗️ Project Architecture

```
├── src/                    # Core GPT-2 Implementation
│   ├── train.py           # Pretraining on FineWeb-Edu
│   ├── model.py           # GPT-2 architecture
│   ├── dataloader.py      # Continuous text loading
│   ├── prepare_dataset.py # FineWeb-Edu dataset prep
│   └── inference.py       # Text generation
│
└── reason/                # Chain-of-Thought Extension
    ├── prepare_gsm8k.py   # Math problem dataset prep
    ├── dataloader.py      # Question-answer loading  
    ├── finetune.py        # Reasoning fine-tuning
    └── inference.py       # Step-by-step problem solving
```

## 🎯 What This Project Does

### Stage 1: Language Model Pretraining
- Trains GPT-2 (124M parameters) from scratch on **FineWeb-Edu** dataset
- Learns general language understanding from 10B educational tokens
- Supports multi-GPU training with PyTorch DDP
- Achieves strong text generation capabilities

### Stage 2: Mathematical Reasoning Extension  
- Fine-tunes the pretrained model on **GSM8K** mathematical word problems
- Learns step-by-step Chain-of-Thought reasoning patterns
- Generates structured solutions: Question → Reasoning → Answer
- Handles complex multi-step mathematical problems

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch numpy tiktoken transformers datasets
```

### 1. Train Base Language Model
```bash
cd src
python prepare_dataset.py    # Download FineWeb-Edu
python train.py              # Train GPT-2 (~46 hours on 2x A100)
```

### 2. Add Mathematical Reasoning
```bash
cd reason
python prepare_gsm8k.py      # Download GSM8K math problems
python finetune.py           # Fine-tune for reasoning (~2 hours)
```

### 3. Test Mathematical Reasoning
```bash
python inference.py --question "Sarah has 15 apples and gives away 7. How many are left?"
```

## 📊 Capabilities & Results

### Base Language Model
- **Training**: 95,365 steps (5 epochs) on FineWeb-Edu
- **Performance**: Strong text generation and completion
- **Architecture**: 12 layers, 768 dimensions, 12 attention heads

### Mathematical Reasoning
- **Dataset**: 7,473 training problems from GSM8K
- **Format**: Natural language word problems requiring multi-step reasoning
- **Output**: Structured step-by-step solutions with numerical answers

### Example Reasoning Output
```
Question: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. How many eggs does she sell?

Let me solve this step by step:
Janet's ducks lay 16 eggs per day.
She eats 3 eggs for breakfast.
She uses 4 eggs to bake muffins.
So she uses a total of 3 + 4 = 7 eggs.
The number of eggs she has left to sell is 16 - 7 = 9.
Therefore, the answer is 9.
```

## 🔧 Technical Implementation

### Modular Design
- **Clean separation**: Base model in `src/`, reasoning extension in `reason/`
- **No code duplication**: Reasoning module imports and extends base model
- **Consistent interfaces**: Both modules follow similar dataloader/training patterns

### Training Details
- **Base model**: Autoregressive language modeling on continuous text
- **Fine-tuning**: Supervised learning on question-answer pairs
- **Optimization**: AdamW with cosine learning rate scheduling
- **Hardware**: CUDA-optimized with automatic mixed precision

### Key Features
- **Tokenization**: GPT-2 BPE tokenizer (50,257 vocabulary)
- **Context length**: 1024 tokens for base model, 512 for reasoning
- **Generation**: Top-k sampling with temperature control
- **Evaluation**: Automatic answer extraction and accuracy measurement

## 📈 Project Highlights

1. **Complete pipeline**: From raw text to mathematical reasoning
2. **Educational focus**: Clean, well-documented code for learning
3. **Extensible design**: Easy to add new reasoning capabilities
4. **Production patterns**: Proper checkpointing, evaluation, and logging

## 🛠️ Development Approach

This project emphasizes **clean, minimal code** over complex features:
- Each file has a single, clear purpose
- Consistent coding patterns across modules  
- Simple command-line interfaces
- Comprehensive but concise implementations

## 📚 Technical References

- **Architecture**: Based on "Attention Is All You Need" (Vaswani et al.)
- **Training**: Inspired by "Language Models are Unsupervised Multitask Learners" (GPT-2 paper)
- **Reasoning**: Chain-of-Thought prompting techniques
- **Dataset**: FineWeb-Edu for pretraining, GSM8K for mathematical reasoning

## 🎓 Educational Value

This implementation serves as a practical example of:
- Large language model training from scratch
- Transfer learning and domain adaptation
- Mathematical reasoning in neural networks
- Clean software architecture for ML projects

Perfect for understanding how modern language models work and how to extend them with specialized capabilities.

## 📄 License

MIT License - feel free to use this code for educational and research purposes.

## 🤝 Contributing

This project is designed for educational purposes. Feel free to fork and experiment with different reasoning capabilities or model architectures.

---
**Note**: Model checkpoints and datasets are not included in this repository due to size constraints. Follow the setup instructions to generate them locally.
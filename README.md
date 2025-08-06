# Vizuara-nanoKimi-Challenge

## nanoKimi: A Minimalist Kimi-K2-Inspired Transformer

This repo contains a single-file PyTorch implementation of a Kimi-K2-style transformer, built for the [Vizuara Research Challenge](https://research-challenge-1.vercel.app/). Inspired by Karpathy’s `nanoGPT`, this project aims for clarity and educational value.

### 🚀 Key Features
- **Latent Attention**: Reduces self-attention complexity to _O(n × L)_ using fixed learnable latent vectors.
- **Mixture of Experts (MoE)**: Replaces FFN with multiple expert networks, gated dynamically per token.

### ⚙️ How to Run
```bash
git clone https://github.com/[YOUR_USERNAME]/nanoKimi-challenge.git
cd nanoKimi-challenge
pip install torch

# Download dataset
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Train model
python train_kimi.py

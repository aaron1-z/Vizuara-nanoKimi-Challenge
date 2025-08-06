import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Hyperparameters ---
# Training
batch_size = 32      # How many independent sequences will we process in parallel?
block_size = 128     # What is the maximum context length for predictions?
max_iters = 5000     # How many training steps to run
eval_interval = 500  # How often to evaluate the model
learning_rate = 3e-4 # Step size for the optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
dropout = 0.1

# Model Architecture
n_embd = 384         # Embedding dimension
n_head = 6           # Number of attention heads
n_layer = 6          # Number of transformer blocks

# Kimi-Specific Hyperparameters
n_latents = 32       # Number of latent vectors for attention (must be << block_size)
num_experts = 8      # Number of experts in the MoE layer
top_k = 2            # Number of experts to route each token to

torch.manual_seed(1337)
# --- End Hyperparameters ---

# --- Data Loading ---
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
# --- End Data Loading ---

# --- nanoKimi Components ---

class Expert(nn.Module):
    """ A single expert in an MoE layer. A simple MLP. """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class MoE(nn.Module):
    """
    A Mixture of Experts layer.
    This is a simplified implementation for educational purposes.
    A more efficient implementation would use sparse operations.
    """
    def __init__(self, n_embd, num_experts, top_k):
        super().__init__()
        self.experts = nn.ModuleList([Expert(n_embd) for _ in range(num_experts)])
        self.gating_network = nn.Linear(n_embd, num_experts)
        self.top_k = top_k

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C) # (B*T, C)
        
        # Get gating logits and select top-k experts
        logits = self.gating_network(x_flat) # (B*T, num_experts)
        gates, indices = torch.topk(logits, self.top_k, dim=-1)
        gates = F.softmax(gates, dim=-1) # (B*T, top_k)
        
        # Combine expert outputs
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            token_indices, expert_indices = torch.where(indices == i)
            
            if token_indices.numel() > 0:
                # Get the corresponding gate values
                gate_values = gates[token_indices, expert_indices].unsqueeze(1)
                # Apply expert and weight by gate
                expert_output = expert(x_flat[token_indices])
                output.index_add_(0, token_indices, expert_output * gate_values)

        return output.view(B, T, C)

class LatentAttention(nn.Module):
    """
    Latent Attention. The model learns a fixed-size set of latent vectors
    that act as a bottleneck for information flow from the sequence.
    This breaks the O(T^2) complexity of standard attention.
    """
    def __init__(self, n_embd, n_head, n_latents):
        super().__init__()
        # Learnable latent array
        self.latents = nn.Parameter(torch.randn(1, n_latents, n_embd))
        # Attention layers
        self.cross_attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)

    def forward(self, x):
        B, T, C = x.shape
        latents = self.latents.expand(B, -1, -1)
        
        # 1. Cross-attention: Latents attend to the input sequence (compress)
        latents, _ = self.cross_attn(query=latents, key=x, value=x)
        # 2. Self-attention: Latents process information internally
        latents, _ = self.self_attn(query=latents, key=latents, value=latents)
        # 3. Cross-attention: Input sequence attends to updated latents (broadcast)
        output, _ = self.cross_attn(query=x, key=latents, value=latents)
        return output



# --- Full nanoKimi Model ---

class KimiBlock(nn.Module):
    """ A nanoKimi Transformer block: Latent Attention followed by MoE """
    def __init__(self, n_embd, n_head, n_latents, num_experts, top_k):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = LatentAttention(n_embd, n_head, n_latents)
        self.ln2 = nn.LayerNorm(n_embd)
        self.moe = MoE(n_embd, num_experts, top_k)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x

class nanoKimi(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[KimiBlock(n_embd, n_head, n_latents, num_experts, top_k) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# --- Training and Optimization ---

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def configure_optimizers(model, weight_decay, learning_rate, betas):
    """
    This function separates the model's parameters into two groups: those that will have
    weight decay applied ('decay') and those that will not ('no_decay').
    This is a common practice for training transformers.
    """
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    # create sets for decay and no_decay parameters
    decay = set()
    no_decay = set()
    
    # define which module types' weights should be decayed
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention) # <-- ADDED MultiheadAttention
    
    # define which module types' weights should NOT be decayed
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

    # iterate over all parameters
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

            if not p.requires_grad:
                continue

            # biases and LayerNorm/Embedding weights should not be decayed
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)
            # weights of Linear and MultiheadAttention layers should be decayed
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            # The custom 'latents' parameters should not be decayed
            elif 'latents' in pn: # <-- ADDED rule for 'latents'
                no_decay.add(fpn)

    # The original script had a bug where it only handled 'blocks.0.attn.latents'.
    # The new rule above handles all of them, so we can remove the old line.
    
    # validate that we considered every parameter
    all_params = decay | no_decay
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer

# Initialize model and optimizer
model = nanoKimi()
m = model.to(device)
print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f} M parameters")
optimizer = configure_optimizers(m, weight_decay=0.1, learning_rate=learning_rate, betas=(0.9, 0.95))

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --- Inference ---
print("\n--- Generating Text ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_text)
print("-----------------------\n")

from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # sequences in parallel
block_size = 8 # context_length
max_iters = 5000
learning_rate = 1e-3
eval_interval = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

###

with open(Path(__file__).parent / 'input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
# In general sub-word tokenisers are used - tiktoken mentioned
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train = data[:n]
val = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train if split == 'train' else val
    # four random offsets into data set
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    # predicting next character
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores -  "affinities"
        # scale by sqrt(head_size), not n_embd, so the softmax isn't too flat
        wei = q @ k.transpose(-2,-1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        # weighted aggregation of the values
        v = self.value(x)
        # perform aggregation
        out = wei @ v # (B, T, T) @ (B, T, C) = (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        # I didn't catch why this projection is necessary
        return self.proj(out)
class FeedForward(nn.Module):
    """ linear layer + NL """

    def __init__(self, n_embd):
        #4x multipliers come from Attention is All you Need
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd),
                                 nn.ReLU(),
                                 # this final one is the same as the projecction above
                                 # he mentions "project back to residual pathway"???
                                 nn.Linear(4 * n_embd, n_embd))
    
    def forward(self, x):
        return self.net(x)
    
# Before we've been doing attention, then ffwd. Lets make that a block
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

    # x = x + self.head(x), means skip connection!
    def forward(self, x):
        x = x + self.heads(x)
        x = x + self.ffwd(x)

        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        #i from xb picks out ith row of token_embedding (logits for next character)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Now use blocks
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
        )
        self.ln_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        # batch, time, channel, (4,8,65)
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.ln_head(x) # (B,T,C)
        # but the Cs are different - one is token embedding, one is vocab size
        # different size info in each

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# Guarded so this module can be imported for its model classes and data helpers
# without training. `python src/zero_to_hero/part_04_gpt/v2.py` trains as before.
if __name__ == "__main__":

    model = BigramLanguageModel()
    m = model.to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for i in range(max_iters):

        xb, yb = get_batch('train')

        if i % eval_interval == 0:
            average_losses = estimate_loss(model)
            print(f"At step {i}: Train loss {average_losses['train']:.4f}, Val Loss {average_losses['val']:.4f}")

        logits, loss = m(xb,yb)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

    print(loss.item())
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

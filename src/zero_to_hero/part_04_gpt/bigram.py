from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
learning_rate = 1e-2
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

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        #i from xb picks out ith row of token_embedding (logits for next character)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.ln_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        # batch, time, channel, (4,8,65)
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
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
            # crop idx to the last block_size tokens - the position table only has block_size rows
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
# without training. `python src/zero_to_hero/part_04_gpt/bigram.py` trains as before.
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

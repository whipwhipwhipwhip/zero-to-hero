"""
N-gram character-level name generator.

Generalises the bigram/trigram single-layer NN from Karpathy's makemore
lecture 2 to an arbitrary context length: predict the next character from
the preceding (gram - 1) characters, using a one-hot-in/softmax-out linear
layer trained with NLL + L2 (equivalent to additively-smoothed counts).
"""

import numpy as np
import argparse
from itertools import product
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

import torch
import torch.nn.functional as F

from ml_learning.utils.parsing import get_vocab

VOCAB = list('.abcdefghijklmnopqrstuvwxyz')


# add custom vocab list
# PyTorchify all of this, and then add all the models I've looked at
class NgramEstimator(BaseEstimator):
    def __init__(self, gram=3, regularisation=0.01, epochs=200, lr=50.0, seed=2147483647, verbose=False, vocab=VOCAB):
        self.gram = gram
        self.regularisation = regularisation
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.verbose = verbose
        self.vocab = vocab


    def fit(self, X, y=None):
        self.W_, self.ctx_stoi_, self.char_stoi_, self.char_itos_ = train_ngram(
            X, gram=self.gram, epochs=self.epochs, lr=self.lr,
            reg=self.regularisation, seed=self.seed, verbose=self.verbose,
        )
        return self

    def score(self, X, y=None):
        return -evaluate_loss(X, self.gram, self.W_, self.ctx_stoi_, self.char_stoi_)


class InterpolatedNgramEstimator(BaseEstimator):
    """Interpolates n-gram models of different orders.

    `gram` is a list of n orders (e.g. [1, 2, 3]); `lam` gives the mixing
    weight of the first n-1 models, with the highest-order model getting
    the remaining weight `1 - sum(lam)`.
    """

    def __init__(self, gram=(2, 3), lam=(0.5,), regularisation=0.01, epochs=200,
                 lr=50.0, seed=2147483647, verbose=False):
        self.gram = gram
        self.lam = lam
        self.regularisation = regularisation
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.verbose = verbose

    def fit(self, X, y=None):
        assert len(self.lam) == len(self.gram) - 1
        self.Ws_, self.ctx_stois_ = [], []
        for g in self.gram:
            W, ctx_stoi, self.char_stoi_, self.char_itos_ = train_ngram(
                X, gram=g, epochs=self.epochs, lr=self.lr,
                reg=self.regularisation, seed=self.seed, verbose=self.verbose,
            )
            self.Ws_.append(W)
            self.ctx_stois_.append(ctx_stoi)
        return self

    def score(self, X, y=None):
        return -evaluate_interpolated_loss(
            X, self.gram, self.Ws_, self.ctx_stois_, self.char_stoi_, self.lam)


def build_context_stoi(gram, vocab=VOCAB):
    """Map every valid (gram - 1)-length context to an index.

    '.' is the start/end token, so it may only appear as a prefix of a
    context (e.g. '..a' is valid, 'a.b' is not — a word can't restart
    mid-context).
    """
    key_length = gram - 1
    stoi = {}
    idx = 0
    for ctx in product(vocab, repeat=key_length):
        dot_positions = [i for i, c in enumerate(ctx) if c == '.']
        if dot_positions and dot_positions[-1] != len(dot_positions) - 1:
            continue
        if key_length == 1:
            ctx = ctx[0]
        stoi[ctx] = idx
        idx += 1
    return stoi


def build_dataset(words, gram, ctx_stoi, char_stoi):
    key_length = gram - 1
    xs, ys = [], []
    for w in words:
        chs = ['.'] * key_length + list(w) + ['.']
        for i in range(len(chs) - key_length):
            ctx = tuple(chs[i:i + key_length])
            nxt = chs[i + key_length]
            if key_length == 1:
                ctx = ctx[0]
            xs.append(ctx_stoi[ctx])
            ys.append(char_stoi[nxt])
    return torch.tensor(xs), torch.tensor(ys)


def train_ngram(words, gram=3, epochs=200, lr=50.0, reg=0.01, seed=2147483647, verbose=False):
    ctx_stoi = build_context_stoi(gram)
    char_stoi = build_context_stoi(2)  # single-character vocab incl. '.'
    char_itos = {i: s for s, i in char_stoi.items()}

    xs, ys = build_dataset(words, gram, ctx_stoi, char_stoi)
    n_contexts = len(ctx_stoi)
    n_chars = len(char_stoi)

    g = torch.Generator().manual_seed(seed)
    W = torch.randn((n_contexts, n_chars), generator=g, requires_grad=True)

    if verbose:
        print(f"training gram={gram} | reg (lambda)={reg}")

    for k in range(epochs):
        logits = W[xs]
        loss = F.cross_entropy(logits, ys) + reg * (W ** 2).mean()

        W.grad = None
        loss.backward()
        W.data += -lr * W.grad

        if verbose and (k % 100 == 0 or k == epochs - 1):
            print(f"epoch {k:4d} | loss {loss.item():.4f}")

    return W, ctx_stoi, char_stoi, char_itos


@torch.no_grad()
def evaluate_loss(words, gram, W, ctx_stoi, char_stoi):
    """NLL loss of a trained W on a (held-out) set of words."""
    xs, ys = build_dataset(words, gram, ctx_stoi, char_stoi)
    logits = W[xs]
    return F.cross_entropy(logits, ys).item()


@torch.no_grad()
def evaluate_interpolated_loss(words, grams, Ws, ctx_stois, char_stoi, lam):
    """NLL loss of a mixture of n-gram models, weighted by `lam` (+ remainder)."""
    weights = list(lam) + [1 - sum(lam)]
    log_ps, ys = [], None
    for g, W, ctx_stoi, w in zip(grams, Ws, ctx_stois, weights):
        xs, ys = build_dataset(words, g, ctx_stoi, char_stoi)
        log_ps.append(torch.log(torch.tensor(w)) + F.log_softmax(W[xs], dim=1))
    log_p_mix = torch.logsumexp(torch.stack(log_ps), dim=0)
    return F.nll_loss(log_p_mix, ys).item()


@torch.no_grad()
def generate(W, ctx_stoi, char_itos, gram, num_samples=10, seed=2147483647, max_length=100):
    key_length = gram - 1
    g = torch.Generator().manual_seed(seed)

    names = []
    for _ in range(num_samples):
        history = ['.'] * key_length
        out = []
        for _ in range(max_length):
            ctx = history[0] if key_length == 1 else tuple(history)
            logits = W[[ctx_stoi[ctx]]]
            p = F.softmax(logits, dim=1)
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            next_char = char_itos[ix]
            if ix == 0:
                break
            out.append(next_char)
            history = (history + [next_char])[-key_length:] if key_length > 0 else []
        names.append(''.join(out))
    return names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="N-gram character-level name generator")
    parser.add_argument('--input-file', '-i', type=str, default='names.txt', help="input file with words, one per line")
    parser.add_argument('--gram', '-g', type=int, default=3, help="context length + 1, e.g. 2=bigram, 3=trigram")
    parser.add_argument('--epochs', type=int, default=200, help="number of gradient descent steps")
    parser.add_argument('--learning-rate', '-l', type=float, default=50.0, help="learning rate")
    parser.add_argument('--reg', type=float, default=0.01, help="L2 regularization strength")
    parser.add_argument('--num-samples', '-n', type=int, default=10, help="number of names to sample")
    parser.add_argument('--test-frac', type=float, default=0.0, help="fraction of words held out to report test loss, e.g. 0.1")
    parser.add_argument('--seed', type=int, default=2147483647, help="seed")
    args = parser.parse_args()

    words = open(args.input_file).read().splitlines()

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(words), generator=g).tolist()
    words = [words[i] for i in perm]
    n_test = int(len(words) * args.test_frac)
    test_words, train_words = words[:n_test], words[n_test:]

    W, ctx_stoi, char_stoi, char_itos = train_ngram(
        train_words, gram=args.gram, epochs=args.epochs, lr=args.learning_rate,
        reg=args.reg, seed=args.seed, verbose=True,
    )

    if test_words:
        test_loss = evaluate_loss(test_words, args.gram, W, ctx_stoi, char_stoi)
        print(f"test loss {test_loss:.4f}")

    print()
    for name in generate(W, ctx_stoi, char_itos, args.gram, num_samples=args.num_samples, seed=args.seed):
        print(name)

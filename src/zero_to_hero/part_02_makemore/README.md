## Part 2 - Makemore

Material from the "Building makemore" lectures, parts 1, 2, 3 and 5 (part 4 is in [Part 3 - Backpropagation](../part_03_backprop)):

1. [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) - bigram character model, first by counting and then as a one-layer neural net.
2. [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I) - the Bengio et al. MLP with learned character embeddings.
3. [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) - initialisation, tanh saturation, and batch normalisation.
5. [Building makemore Part 5: Building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) - a hierarchical model that fuses characters in pairs, following WaveNet.

Every model is trained on [names.txt](names.txt), a list of ~32k names, and generates new name-like strings one character at a time.

### What's here

- [makemore.py](makemore.py) - generalises the part 1 bigram neural net to any n-gram length: a one-hot context goes in, a softmax over the next character comes out, trained with NLL + L2 regularisation. It includes:
  - `NgramEstimator` and `InterpolatedNgramEstimator` - sklearn-compatible wrappers, so the models can go through `GridSearchCV` and train/test splits. The interpolated version mixes n-gram models of different orders.
  - a command-line interface for training a model and sampling names.

```bash
uv run python src/zero_to_hero/part_02_makemore/makemore.py -i src/zero_to_hero/part_02_makemore/names.txt --gram 3 --test-frac 0.1
```

### Notebooks

In [notebooks/02_makemore](../../../notebooks/02_makemore), numbered by lecture order in the playlist (micrograd is lecture 1):

- `lecture_02.ipynb` - makemore part 1: bigram counts, sampling, and the equivalent neural net.
- `lecture_03.ipynb` - makemore part 2: the MLP.
- `lecture_04.ipynb`, `lecture_04-bn.ipynb`, `lecture_04_pytorch.ipynb` - makemore part 3: fixing the initial loss and tanh saturation, batch normalisation (including estimating the running mean/std), and the same network rebuilt with PyTorch-style layer classes plus activation and gradient histograms.
- `lecture_05.ipynb` - makemore part 5: the WaveNet-style model.

Exercises, in folders matching the lecture numbers:

- `exercises_02/` - part 1 exercises: a trigram model, train/dev/test splits, tuning the regularisation strength, and a grid search over hyperparameters (including interpolated n-grams). These use `makemore.py`.
- `exercises_03/` - part 2 exercises: tuning the MLP with minibatches, He/Glorot initialisation, and adding skip connections.
- `exercises_04/` - part 3 exercises: what happens with zero-initialised weights, and folding batchnorm parameters into the preceding linear layer after training.
- `exercises_05/dilations.ipynb` - part 5: the WaveNet model with dilations, plus a simple experiment harness.

### Tests

From the repo root:

```bash
uv run pytest tests/makemore
```

The tests check the context indexing and dataset construction, that training reduces the loss, that sampling works, and that the estimators work with `GridSearchCV`.

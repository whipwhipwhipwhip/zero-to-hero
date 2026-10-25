import pytest
import torch
from sklearn.model_selection import GridSearchCV

from ml_learning.makemore_conor.makemore import (
    NgramEstimator,
    build_context_stoi,
    build_dataset,
    evaluate_loss,
    generate,
    train_ngram,
)

WORDS = ["ana", "anna", "ann", "ben", "bo"]


def test_build_context_stoi_bigram_is_full_alphabet_plus_dot():
    stoi = build_context_stoi(2)
    assert len(stoi) == 27
    assert stoi["."] == 0


def test_build_context_stoi_trigram_excludes_invalid_dot_placements():
    stoi = build_context_stoi(3)
    # 26*26 with no dot, + 26 with a single leading dot, + 1 for "..": matches
    # the Lecture 2 trigram count table size.
    assert len(stoi) == 703
    assert ('.', '.') in stoi
    assert ('.', 'a') in stoi
    assert ('a', '.') not in stoi  # dot can't appear after a real character


def test_build_dataset_produces_expected_context_target_pairs():
    ctx_stoi = build_context_stoi(3)
    char_stoi = build_context_stoi(2)
    xs, ys = build_dataset(["ab"], gram=3, ctx_stoi=ctx_stoi, char_stoi=char_stoi)

    expected_contexts = [('.', '.'), ('.', 'a'), ('a', 'b')]
    expected_targets = ['a', 'b', '.']
    assert xs.tolist() == [ctx_stoi[c] for c in expected_contexts]
    assert ys.tolist() == [char_stoi[c] for c in expected_targets]


@pytest.mark.parametrize("gram", [2, 3])
def test_train_ngram_reduces_loss_over_epochs(gram):
    # lr=50 (tuned for the full names.txt corpus) diverges on this 5-word
    # toy set, so use a lower lr here.
    torch.manual_seed(0)
    W, ctx_stoi, char_stoi, char_itos = train_ngram(WORDS, gram=gram, epochs=1, lr=10.0, reg=0.01)
    loss_after_one_epoch = evaluate_loss(WORDS, gram, W, ctx_stoi, char_stoi)

    W2, ctx_stoi2, char_stoi2, char_itos2 = train_ngram(WORDS, gram=gram, epochs=200, lr=10.0, reg=0.01)
    loss_after_many_epochs = evaluate_loss(WORDS, gram, W2, ctx_stoi2, char_stoi2)

    assert loss_after_many_epochs < loss_after_one_epoch


def test_evaluate_loss_is_finite_on_unseen_words():
    W, ctx_stoi, char_stoi, char_itos = train_ngram(WORDS, gram=3, epochs=50)
    loss = evaluate_loss(["zara"], 3, W, ctx_stoi, char_stoi)
    assert loss == pytest.approx(loss)  # finite, not nan/inf
    assert loss > 0


@pytest.mark.parametrize("gram", [1, 2, 3])
def test_generate_produces_requested_number_of_strings(gram):
    W, ctx_stoi, char_stoi, char_itos = train_ngram(WORDS, gram=gram, epochs=10)
    names = generate(W, ctx_stoi, char_itos, gram, num_samples=4, seed=0)
    assert len(names) == 4
    assert all(isinstance(name, str) for name in names)


def test_ngram_estimator_fit_and_score():
    est = NgramEstimator(gram=3, regularisation=0.1, epochs=50)
    est.fit(WORDS)
    score = est.score(WORDS)
    assert score < 0  # negative NLL loss


def test_ngram_estimator_works_with_grid_search_cv():
    gs = GridSearchCV(
        NgramEstimator(gram=2, epochs=20),
        param_grid={"regularisation": [0.01, 1.0]},
        cv=2,
    )
    gs.fit(WORDS * 4)  # repeat so each CV fold has enough examples
    assert "regularisation" in gs.best_params_

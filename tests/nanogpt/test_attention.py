"""`model.attention`: the two implementations, and that they agree.

The reason this module exists at all is that the manual path stopped being
reachable when `self.flash` was a `hasattr` probe — always true on torch 2.x, so
the `else` branch was dead code. Making it a config choice is only worth doing if
something checks the two paths still compute the same function, which is what the
equivalence tests here are for.

Cheap tier: CPU, tiny model, no filesystem, no training. Nothing here asserts
which is *faster* — that's hardware, and on MPS the answer is currently "neither".
"""

from __future__ import annotations

import pytest
import torch

from ml_learning.nanogpt_conor.config import ATTENTIONS, GPTConfig
from ml_learning.nanogpt_conor.model import GPTModel, load_state_dict_compat

VOCAB = 17


def cfg(**overrides) -> GPTConfig:
    base = dict(n_embd=32, n_head=4, n_layer=2, block_size=16, dropout=0.0)
    base.update(overrides)
    return GPTConfig(**base)


def build(attention: str, seed: int = 0) -> GPTModel:
    """Same seed, so two models differ only in how they attend."""
    torch.manual_seed(seed)
    model = GPTModel(cfg(attention=attention), VOCAB)
    return model.eval()  # dropout off; the paths apply it in different places


# ---- the config knob ------------------------------------------------------

def test_the_default_is_sdpa():
    assert cfg().attention == "sdpa"


def test_an_unknown_implementation_is_rejected():
    """Beside the config that named it, not as an AttributeError in a forward."""
    with pytest.raises(ValueError, match="unknown attention"):
        cfg(attention="flash")


def test_both_registered_implementations_build():
    for attention in ATTENTIONS:
        assert build(attention) is not None


def test_only_the_manual_path_holds_a_mask():
    """sdpa gets causality from `is_causal=True`; manual needs the buffer."""
    assert not hasattr(build("sdpa").blocks[0].heads, "tril")
    assert hasattr(build("manual").blocks[0].heads, "tril")


# ---- checkpoint shape -----------------------------------------------------

def test_the_mask_is_not_checkpointed():
    """persistent=False. It's derived from block_size, not learned."""
    assert not [k for k in build("manual").state_dict() if k.endswith("tril")]


def test_both_implementations_checkpoint_identically():
    """The payoff: switching implementation doesn't invalidate a checkpoint."""
    assert list(build("sdpa").state_dict()) == list(build("manual").state_dict())


def test_a_checkpoint_moves_between_implementations():
    sdpa, manual = build("sdpa"), build("manual", seed=1)
    manual.load_state_dict(sdpa.state_dict())  # strict: no missing, no unexpected

    x = torch.randint(0, VOCAB, (2, 8))
    assert torch.allclose(sdpa(x)[0], manual(x)[0], atol=1e-5)


# ---- the two paths agree --------------------------------------------------

def test_logits_match():
    """If these diverge, one of the two is wrong and the config is a trap."""
    sdpa, manual = build("sdpa"), build("manual")
    x = torch.randint(0, VOCAB, (4, 12))
    assert torch.allclose(sdpa(x)[0], manual(x)[0], atol=1e-5)


def test_loss_matches():
    sdpa, manual = build("sdpa"), build("manual")
    x = torch.randint(0, VOCAB, (4, 12))
    y = torch.randint(0, VOCAB, (4, 12))
    assert sdpa(x, y)[1].item() == pytest.approx(manual(x, y)[1].item(), abs=1e-5)


def test_gradients_match():
    """Forward agreement isn't enough — the two paths have different backwards."""
    x = torch.randint(0, VOCAB, (4, 12))
    y = torch.randint(0, VOCAB, (4, 12))

    grads = {}
    for attention in ATTENTIONS:
        model = build(attention)
        model(x, y)[1].backward()
        grads[attention] = model.blocks[0].heads.query.weight.grad

    assert torch.allclose(grads["sdpa"], grads["manual"], atol=1e-5)


def test_attention_is_causal_in_both():
    """A token's logits must not move when a *later* token changes."""
    for attention in ATTENTIONS:
        model = build(attention)
        x = torch.randint(0, VOCAB, (1, 8))
        before = model(x)[0][0, 3].clone()
        x[0, 7] = (x[0, 7] + 1) % VOCAB
        assert torch.allclose(before, model(x)[0][0, 3], atol=1e-6), attention


# ---- loading checkpoints written before any of this ------------------------

def test_legacy_tril_keys_are_dropped():
    """Checkpoints from before `tril` stopped being persistent carry one per
    block; they're a constant mask, so dropping them loses nothing."""
    model = build("sdpa")
    legacy = dict(model.state_dict())
    for i in range(2):
        legacy[f"blocks.{i}.heads.tril"] = torch.tril(torch.ones(1, 1, 16, 16))

    with pytest.raises(RuntimeError, match="Unexpected key"):
        model.load_state_dict(legacy)

    load_state_dict_compat(model, legacy)  # not raising is the assertion


def test_a_genuinely_missing_weight_still_fails():
    """Why this isn't just `strict=False`: that would swallow a corrupt
    checkpoint as happily as an old one."""
    model = build("sdpa")
    broken = dict(model.state_dict())
    del broken["blocks.0.heads.query.weight"]

    with pytest.raises(RuntimeError, match="Missing key"):
        load_state_dict_compat(model, broken)

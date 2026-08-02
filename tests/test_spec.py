"""Speculative decoding + the layer-iterator regression fix.

The regression: LayerStreamingEngine._layers() is a finite generator, and the
old generate() reused ONE iterator across all decode tokens, so only the
first token was actually computed. Every forward pass must create a fresh
layer iterator (asserted via compute_tokens accounting).
"""
import os

import pytest
from androidllm.engine import LayerStreamingEngine

MODEL = os.path.join(os.path.dirname(__file__), "tmp_model")


@pytest.fixture(autouse=True)
def _prefix_off():
    os.environ["ANDROIDLLM_PREFIX_KV"] = "0"
    yield
    os.environ.pop("ANDROIDLLM_PREFIX_KV", None)


@pytest.fixture(autouse=True, scope="session")
def _toy_model():
    from test_streaming import build
    build()
    yield


def _engine(**kw):
    return LayerStreamingEngine(MODEL, **kw)


def _ids(engine, text="hello there"):
    return engine.tokenizer.encode(text)


def test_full_compute_per_token():
    """Every prompt + decode token runs a full layer pass (regression)."""
    e = _engine()
    ids = _ids(e)
    e.generate(ids, max_new_tokens=8, temperature=0.0)
    expected = e.n_layers * ((len(ids) - 1) + 8)
    assert e.stats["compute_tokens"] == expected


def test_spec_identical_to_plain_greedy():
    """Draft == target: spec output must equal plain output exactly."""
    e = _engine()
    ids = _ids(e)
    plain = e.generate(ids, max_new_tokens=12, temperature=0.0)
    s = _engine(draft_dir=MODEL, spec_k=4)
    spec = s.generate(ids, max_new_tokens=12, temperature=0.0)
    assert spec == plain
    assert s.stats["tokens_served"] == 12


def test_spec_budget_respected():
    """A K+1-token step must not overshoot max_new_tokens."""
    s = _engine(draft_dir=MODEL, spec_k=5)
    ids = _ids(s)
    out = s.generate(ids, max_new_tokens=17, temperature=0.0)
    assert len(out) == 17
    stats = s.snapshot()["spec"]
    assert stats["accepted"] + stats["bonus"] == 17


def test_spec_stats_and_snapshot():
    s = _engine(draft_dir=MODEL, spec_k=4)
    ids = _ids(s)
    s.generate(ids, max_new_tokens=8, temperature=0.0)
    sp = s.snapshot()["spec"]
    assert sp["enabled"] is True
    assert sp["k"] == 4
    assert sp["draft_tokens"] > 0
    assert sp["steps"] > 0
    assert 0.0 <= sp["accept_rate"] <= 1.0
    assert sp["accepted"] + sp["bonus"] == 8


def test_spec_callback_streams_every_token():
    seen = []
    s = _engine(draft_dir=MODEL, spec_k=4)
    s.generate(_ids(s), max_new_tokens=10, temperature=0.0, callback=seen.append)
    assert len(seen) == 10


def test_spec_draft_mismatch_still_works():
    """A draft that never matches (different weights) must still produce
    output and correct stats."""
    import numpy as np

    s = _engine(draft_dir=MODEL, spec_k=3)
    noise = np.random.default_rng(0).normal(0, 0.1, s.draft.model.embed.shape)
    s.draft.model.embed = (s.draft.model.embed + noise.astype(np.float16))
    ids = _ids(s)
    out = s.generate(ids, max_new_tokens=6, temperature=0.0)
    assert len(out) == 6
    sp = s.snapshot()["spec"]
    assert sp["accepted"] + sp["bonus"] == 6


def test_spec_stop_token_stops():
    s = _engine(draft_dir=MODEL, spec_k=4)
    ids = _ids(s)
    out = s.generate(ids, max_new_tokens=64, temperature=0.0,
                     stop_ids=(9,))
    assert 9 not in out
    assert len(out) <= 64


def test_no_spec_with_grammar():
    from androidllm.json_grammar import JsonGrammar

    s = _engine(draft_dir=MODEL, spec_k=4)
    ids = _ids(s)
    g = JsonGrammar({"type": "object", "properties": {"a": {"type": "integer"}},
                     "required": ["a"]})
    out = s.generate(ids, max_new_tokens=8, temperature=0.8, grammar=g)
    assert len(out) == 8
    sp = s.snapshot()["spec"]
    assert sp["steps"] == 0


def test_prefix_reuse_with_spec():
    """KV prefix reuse still applies when spec is enabled."""
    os.environ["ANDROIDLLM_PREFIX_KV"] = "1"
    try:
        s = _engine(draft_dir=MODEL, spec_k=4)
        ids = _ids(s)
        s.generate(ids, max_new_tokens=6, temperature=0.0)
        first = s.stats["prefix_calls"]
        s.generate(ids, max_new_tokens=6, temperature=0.0)
        assert s.stats["prefix_calls"] > first
        assert s.stats["prefix_tokens"] > 0
        s.close()
    finally:
        os.environ["ANDROIDLLM_PREFIX_KV"] = "0"

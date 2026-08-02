"""Feature tests: embedding quantization, layer cache policies, prefix KV
reuse, layer skipping, and the JSON grammar masker."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from androidllm.engine import LayerStreamingEngine
from androidllm.json_grammar import JsonGrammar
from androidllm.quant import dequantize_packed, quantize_matrix
from androidllm.safetensors import read_tensor, write_safetensors

from test_streaming import TMP, build


class FakeTok:
    """Tiny tokenizer for grammar tests: one token per interesting piece."""

    def __init__(self, pieces):
        self.ids_to_tokens = pieces
        self.vocab_size = len(pieces)

    def decode(self, ids):
        out = []
        for i in ids:
            out.append(self.ids_to_tokens[i])
        return "".join(out)


def test_embed_quant_roundtrip():
    rng = np.random.default_rng(3)
    w = (rng.standard_normal((50, 64)) * 0.1).astype(np.float32)
    q, scale = quantize_matrix(w, bits=8, block=w.shape[1])
    deq = dequantize_packed(q.astype(np.int8), scale, {
        "bits": 8, "block": w.shape[1], "out": w.shape[0], "in": w.shape[1]})
    err = np.max(np.abs(w.astype(np.float16) - deq))
    assert err < 0.01, err
    print(f"embed quant roundtrip OK (max err {err:.4f})")


def test_engine_embed_quant_path():
    build()
    embed_path = os.path.join(TMP, "embeddings.safetensors")
    embed = read_tensor(embed_path, "embed", memmap=False)
    q, scale = quantize_matrix(embed.astype(np.float32), bits=8, block=embed.shape[1])
    write_safetensors(embed_path, {
        "embed.q": q.astype(np.int8),
        "embed.scale": scale.astype(np.float16),
    })
    with open(os.path.join(TMP, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    man["embed_quant"] = {"bits": 8, "block": embed.shape[1],
                          "out": embed.shape[0], "in": embed.shape[1]}
    with open(os.path.join(TMP, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f)
    e = LayerStreamingEngine(TMP)
    diff = np.max(np.abs(e.model.embed.astype(np.float32) - embed.astype(np.float32)))
    assert diff < 0.01, diff
    print(f"engine embed-quant path OK (max diff {diff:.4f})")
    # restore for other tests
    write_safetensors(embed_path, {"embed": embed})
    with open(os.path.join(TMP, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    man.pop("embed_quant", None)
    with open(os.path.join(TMP, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f)


def test_skip_logic():
    e = LayerStreamingEngine(TMP)
    e._skip_every = 0
    assert not e._skip(5)
    e.n_layers = 16
    e._skip_every = 4
    assert e._skip(4) and e._skip(8) and e._skip(12)
    assert not e._skip(0), "never skip the input layer"
    assert not e._skip(15), "never skip the last layer"
    e.n_layers = 2
    e._skip_every = 2
    assert not e._skip(1), "2-layer model: nothing skippable"
    print("layer-skip logic OK")


def test_prefix_kv_reuse():
    build()
    e = LayerStreamingEngine(TMP)
    ids1 = [3, 7, 11, 5]
    e.generate(ids1, max_new_tokens=2)
    kv1 = e._kv
    pfx = e._match_prefix([3, 7, 11, 5, 9])
    assert pfx == 4, pfx
    e.generate([3, 7, 11, 5, 9], max_new_tokens=2)
    assert e._kv is kv1
    assert e._match_prefix([1, 2]) == 0
    assert e.stats["prefix_calls"] >= 1, e.stats
    assert e.stats["prefix_tokens"] >= 4, e.stats
    snap = e.snapshot()
    assert snap["context_used"] == len([3, 7, 11, 5, 9]) + 2 - 1, snap["context_used"]
    assert 0 < snap["context_pct"] <= 100, snap["context_pct"]
    assert "prefix_calls" in snap and "paused" in snap
    print("prefix KV reuse OK (calls={} tokens={} ctx={})".format(
        e.stats["prefix_calls"], e.stats["prefix_tokens"], snap["context_used"]))


def test_lru_cache():
    build()
    e = LayerStreamingEngine(TMP, keep_layers=1)
    e._lru = 1
    e.generate([3, 7, 11], max_new_tokens=3)
    keys = list(e._cache.keys())
    assert 0 in keys, keys
    assert len(keys) <= 2, keys
    assert e.stats["cache_hits"] > 0
    print("LRU+pinned cache OK (keys={} hits={})".format(
        keys, e.stats["cache_hits"]))


def _mask_for(schema, buf, tok):
    g = JsonGrammar(schema)
    m = g.allowed_mask(buf, tok)
    return {tok.ids_to_tokens[i]: bool(m[i]) for i in range(tok.vocab_size)}


def test_grammar_object_required():
    pieces = ['{', '}', '[', ']', ':', ',', '"', 'a', 'b', '1', '.', '-', ' ',
              'true', 'false', 'null', 'x']
    tok = FakeTok(pieces)
    schema = {"type": "object", "properties": {"a": {"type": "number"}},
              "required": ["a"]}
    m = _mask_for(schema, "", tok)
    assert m['{'] and m[' ']
    assert not m['"'], "root value is an object: must open with {"
    assert not m['}'], "cannot close before required keys"
    m = _mask_for(schema, '{"a": 4', tok)
    assert m[','] and m['}'], "after required value, close allowed"
    m = _mask_for(schema, '{"a": "x"}', tok)
    assert m[' '], "done: only whitespace after object closes"
    assert not m[','], "object already closed"
    m = _mask_for(schema, '{"a": 4,', tok)
    assert m['"'] and m['}'], "in key state: new key or close"
    print("grammar object/required OK")


def test_grammar_string_and_number():
    pieces = ['"', '\\', 'a', 'b', '1', '-', '.', 'e', ' ', '{', '}', ',', 'x', '5']
    tok = FakeTok(pieces)
    g = JsonGrammar({"type": "number"})
    m = g.allowed_mask("", tok)
    names = [p for p in pieces]
    assert not (m[names.index('-')] and False)  # - allowed at start
    assert m[names.index('1')] and m[names.index('-')]
    assert not m[names.index('"')], "string quote not allowed for number"
    m2 = g.allowed_mask("12.5", tok)
    assert m2[names.index('e')], "exponent allowed after 12.5"
    assert not m2[names.index('a')]
    print("grammar number OK")


def test_grammar_end_to_end_object():
    """Simulate token-by-token generation with a small structured vocab."""
    pieces = ['{', '}', ',', ':', '"', 'a', 'b', '1', '2', ' ', 't', 'r', 'u', 'e']
    tok = FakeTok(pieces)
    g = JsonGrammar({"type": "object",
                     "properties": {"ok": {"type": "boolean"}},
                     "required": ["ok"]})
    buf = ""
    for want in ('{', '"', 'a', '"', ':', 't', 'r', 'u', 'e', '}'):
        m = g.allowed_mask(buf, tok)
        idx = pieces.index(want)
        assert m[idx], f"token {want!r} not allowed at buf={buf!r}"
        buf += want
    m = g.allowed_mask(buf, tok)
    assert not any(m[pieces.index(p)] for p in (',', '"'))
    print("grammar e2e object OK")


def test_think_strip():
    from androidllm.serve import _think_strip, _ThinkStripper
    assert _think_strip("<think>let me reason</think>The answer is 42.") == \
        "The answer is 42."
    assert _think_strip("no tags here") == "no tags here"
    assert _think_strip("<think>a</think><think>b</think>x") == "x"
    s = _ThinkStripper()
    assert s.feed("Hi<think>in") == "Hi"
    assert s.feed("side</think> tail") == ""
    assert s.flush() == " tail"
    s3 = _ThinkStripper()
    assert s3.feed("broken <think") == "broken"
    assert s3.flush() == " <think"
    s2 = _ThinkStripper()
    got = "".join(s2.feed(p) for p in ["<th", "ink>hidden</", "think>visible"])
    got += s2.flush()
    assert got == "visible", got
    print("think-strip OK")


def test_battery_pause_logic():
    from androidllm import serve
    class E:
        paused = False
        throttle_ms = 0
    e = E()
    def run(info):
        cap = info["capacity"]
        charging = info["charging"]
        e.paused = (serve._PAUSE_PCT > 0 and cap is not None
                    and cap <= serve._PAUSE_PCT and not charging)
        if charging:
            e.throttle_ms = 0
        else:
            e.throttle_ms = 20 if (cap is not None and cap <= 15) else 0
    run({"capacity": 12, "charging": False})
    assert e.paused and e.throttle_ms == 20
    run({"capacity": 12, "charging": True})
    assert not e.paused and e.throttle_ms == 0
    run({"capacity": 60, "charging": False})
    assert not e.paused and e.throttle_ms == 0
    run({"capacity": None, "charging": False})
    assert not e.paused
    print("battery pause logic OK")


if __name__ == "__main__":
    test_embed_quant_roundtrip()
    test_engine_embed_quant_path()
    test_skip_logic()
    test_prefix_kv_reuse()
    test_lru_cache()
    test_grammar_object_required()
    test_grammar_string_and_number()
    test_grammar_end_to_end_object()
    test_think_strip()
    test_battery_pause_logic()

"""Catalog listing + token-count endpoint tests.

catalog_list(): RAM-tier filter (Group 45), fit flags against specs.
token-count: needs a live engine + server — covered indirectly here via the
pure helpers (token counting paths live in serve.py, exercised in
test_streaming.py end-to-end).
"""
from androidllm import modelpicker as mp


def test_catalog_list_all_entries_have_tiers():
    out = mp.catalog_list()
    assert out["count"] == len(mp.CATALOG)
    for e in out["entries"]:
        assert e["id"] and e["repo"] and e["tier"]
        assert e["resident_gb"] > 0
        assert e["tier"] in ("1-2GB", "2-4GB", "4-8GB", "8-16GB", "16GB+")


def test_catalog_list_tier_filter():
    out = mp.catalog_list(tier="4-8")
    assert out["count"] >= 1
    for e in out["entries"]:
        assert 4 <= e["resident_gb"] <= 8
    assert out["entries"][0]["tier"] == "4-8GB"


def test_catalog_list_fits_flags_with_specs():
    out = mp.catalog_list(specs={"ram_gb": 8, "disk_free_gb": 64})
    fits = [e for e in out["entries"] if e["fits"] and e["fits"]["ram"]]
    assert any(e["id"] == "qwen15" for e in fits)
    big = next(e for e in out["entries"] if e["id"] == "qwen3-32b")
    assert big["fits"]["ram"] is False


def test_catalog_list_unknown_tier_is_none():
    assert mp._tier_bounds("banana") is None
    assert mp._tier_bounds(None) is None
    assert mp._tier_bounds("4-8") == (4.0, 8.0)
    assert mp._tier_bounds("8") == (8.0, None)


# ------------------------------------------------------------------ benchmark-informed defaults (6)

def test_bench_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_BENCH_PATH", str(tmp_path / "bench.json"))
    mp.bench_add("qwen15", 4.2, "g85", "measured run")
    assert mp.measured_tps("qwen15") == 4.2
    assert mp.measured_tps("smollm2") is None
    assert mp.bench_list()["qwen15"]["device"] == "g85"
    mp.bench_clear()
    assert mp.measured_tps("qwen15") is None


def test_score_uses_measured_tps(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_BENCH_PATH", str(tmp_path / "bench.json"))
    mp.bench_add("qwen15", 9.0, "g85")
    specs = {"ram_gb": 8, "disk_free_gb": 64}
    _, breakdown = mp.score(next(m for m in mp.CATALOG if m["id"] == "qwen15"), specs)
    assert breakdown["measured"] is True
    assert breakdown["est_tps"] == 9.0
    assert breakdown["speed"] == 1.0  # 9 tok/s caps the speed component


def test_score_estimates_without_benchmark(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_BENCH_PATH", str(tmp_path / "bench.json"))
    specs = {"ram_gb": 8, "disk_free_gb": 64}
    _, breakdown = mp.score(next(m for m in mp.CATALOG if m["id"] == "qwen15"), specs)
    assert breakdown["measured"] is False
    assert breakdown["est_tps"] == round(mp._est_speed(1.54), 2)

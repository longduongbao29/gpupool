import os

from gpupool.agent.rpc_cache import prune, rpc_cache_dir


def _file(d, name, size, age):
    p = d / name
    p.write_bytes(b"x" * size)
    t = 1_000_000 - age
    os.utime(p, (t, t))
    return p


def test_prune_deletes_least_recently_used_until_under_the_cap(tmp_path):
    old = _file(tmp_path, "a", 400, age=300)
    mid = _file(tmp_path, "b", 400, age=200)
    new = _file(tmp_path, "c", 400, age=100)
    assert prune(tmp_path, 900) == ["a"]
    assert not old.exists() and mid.exists() and new.exists()
    assert prune(tmp_path, 900) == []  # already under the cap
    assert prune(tmp_path, 0) == []  # 0: no cap


def test_prune_missing_dir(tmp_path):
    assert prune(tmp_path / "nope", 10) == []


def test_cache_dir_follows_llama_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_CACHE", raising=False)
    assert rpc_cache_dir(tmp_path) == tmp_path / "rpc"
    monkeypatch.setenv("LLAMA_CACHE", "/elsewhere")
    assert str(rpc_cache_dir(tmp_path)) == "/elsewhere/rpc"

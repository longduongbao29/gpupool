"""Model library: HF listing/download (respx-mocked), path registration, delete, /files."""
from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gpupool.common.auth import require_bearer
from gpupool.coordinator import library as library_mod
from gpupool.coordinator.library import Library, LibraryError
from gpupool.coordinator.library_api import make_files_router, make_library_router

HF = "https://huggingface.co"
REPO = "acme/model-GGUF"


def tree(*entries):
    return [{"type": t, "path": p, "size": s} for t, p, s in entries]


@pytest.fixture
def lib(tmp_path):
    http = httpx.AsyncClient(follow_redirects=True)
    lb = Library(":memory:", tmp_path / "models", hf_token="", http=http)
    yield lb
    lb._db.close()


async def wait_status(lib, name, status, timeout=5.0):
    for _ in range(int(timeout / 0.01)):
        it = lib.get(name)
        if it and it.status == status:
            return it
        await asyncio.sleep(0.01)
    raise AssertionError(f"{name} never reached {status}: {lib.get(name)}")


# ---- listing -----------------------------------------------------------------------

@respx.mock
async def test_hf_files_filters_and_sorts(lib):
    route = respx.get(f"{HF}/api/models/{REPO}/tree/main").respond(json=tree(
        ("file", "b.gguf", 20), ("file", "README.md", 1), ("directory", "x.gguf", 0),
        ("file", "sub/a.gguf", 10)))
    assert await lib.hf_files(REPO) == [{"file": "b.gguf", "bytes": 20},
                                        {"file": "sub/a.gguf", "bytes": 10}]
    assert route.calls[0].request.url.params["recursive"] == "true"


@respx.mock
async def test_hf_files_sends_token(tmp_path):
    lb = Library(":memory:", tmp_path, hf_token="tok", http=httpx.AsyncClient())
    route = respx.get(f"{HF}/api/models/{REPO}/tree/main").respond(json=[])
    await lb.hf_files(REPO)
    assert route.calls[0].request.headers["authorization"] == "Bearer tok"


@respx.mock
@pytest.mark.parametrize("code,msg,status", [
    (401, "gated or private: set HF_TOKEN", 403), (403, "gated or private: set HF_TOKEN", 403),
    (404, "repo not found", 404)])
async def test_hf_files_errors(lib, code, msg, status):
    respx.get(f"{HF}/api/models/{REPO}/tree/main").respond(code)
    with pytest.raises(LibraryError, match=msg) as ei:
        await lib.hf_files(REPO)
    assert ei.value.status == status


@pytest.mark.parametrize("repo", ["noslash", "a/b/c", "../x", "a/..", "a b/c", ""])
async def test_hf_repo_validation(lib, repo):
    with pytest.raises(LibraryError):
        await lib.hf_files(repo)


# ---- downloads ---------------------------------------------------------------------

@respx.mock
async def test_download_happy_path(lib, monkeypatch):
    monkeypatch.setattr(library_mod, "_PROGRESS_INTERVAL_S", 0.0)
    body = b"x" * (3 * 1024 * 1024)
    respx.get(f"{HF}/{REPO}/resolve/main/sub/m.gguf").respond(content=body)
    item = await lib.add_hf(REPO, "sub/m.gguf")
    assert item.name == "m.gguf" and item.status == "downloading" and item.source == "hf"
    done = await wait_status(lib, "m.gguf", "ready")
    assert done.downloaded == len(body) == done.bytes
    assert (lib.models_dir / "m.gguf").read_bytes() == body
    assert not list(lib.models_dir.glob("*.part"))
    assert lib.resolve("m.gguf") == (lib.models_dir / "m.gguf").resolve()


@respx.mock
async def test_progress_is_written_while_streaming(lib, monkeypatch):
    monkeypatch.setattr(library_mod, "_PROGRESS_INTERVAL_S", 0.0)
    release = asyncio.Event()

    async def stream():
        yield b"a" * (1024 * 1024)
        yield b"b" * (1024 * 1024)
        await release.wait()
        yield b"c"

    respx.get(f"{HF}/{REPO}/resolve/main/m.gguf").mock(
        return_value=httpx.Response(200, stream=stream()))
    await lib.add_hf(REPO, "m.gguf")
    for _ in range(500):
        if lib.get("m.gguf").downloaded >= 1024 * 1024:
            break
        await asyncio.sleep(0.01)
    assert lib.get("m.gguf").downloaded >= 1024 * 1024
    assert lib.get("m.gguf").status == "downloading"
    assert lib.resolve("m.gguf") is None  # not ready yet
    release.set()
    await wait_status(lib, "m.gguf", "ready")


@respx.mock
async def test_size_mismatch_fails_and_leaves_no_part(lib):
    respx.get(f"{HF}/{REPO}/resolve/main/m.gguf").respond(
        content=b"abc", headers={"content-length": "10"})
    # httpx may refuse the short body itself; either way the item must end failed.
    await lib.add_hf(REPO, "m.gguf")
    it = await wait_status(lib, "m.gguf", "failed")
    assert it.error
    assert not list(lib.models_dir.iterdir())
    assert lib.resolve("m.gguf") is None


@respx.mock
async def test_listing_size_mismatch_for_split_fails(lib):
    respx.get(f"{HF}/api/models/{REPO}/tree/main").respond(json=tree(
        ("file", "m-00001-of-00002.gguf", 5), ("file", "m-00002-of-00002.gguf", 5)))
    respx.get(f"{HF}/{REPO}/resolve/main/m-00001-of-00002.gguf").respond(
        content=b"12345", headers={"content-length": "5"})
    respx.get(f"{HF}/{REPO}/resolve/main/m-00002-of-00002.gguf").respond(content=b"123")
    await lib.add_hf(REPO, "m-00001-of-00002.gguf")
    it = await wait_status(lib, "m-00001-of-00002.gguf", "failed")
    assert "truncated" in it.error
    # Even the part that completed must not stay behind for a failed item.
    assert not list(lib.models_dir.iterdir())


@respx.mock
async def test_http_error_marks_failed(lib):
    respx.get(f"{HF}/{REPO}/resolve/main/m.gguf").respond(403)
    await lib.add_hf(REPO, "m.gguf")
    it = await wait_status(lib, "m.gguf", "failed")
    assert "HF_TOKEN" in it.error


@respx.mock
async def test_split_download_all_parts(lib):
    respx.get(f"{HF}/api/models/{REPO}/tree/main").respond(json=tree(
        ("file", "q/m-00001-of-00003.gguf", 4), ("file", "q/m-00002-of-00003.gguf", 5),
        ("file", "q/m-00003-of-00003.gguf", 6), ("file", "q/other.gguf", 99)))
    for i, n in enumerate((4, 5, 6), start=1):
        respx.get(f"{HF}/{REPO}/resolve/main/q/m-0000{i}-of-00003.gguf").respond(content=b"z" * n)
    item = await lib.add_hf(REPO, "q/m-00001-of-00003.gguf")
    assert item.name == "m-00001-of-00003.gguf" and item.bytes == 15
    done = await wait_status(lib, item.name, "ready")
    assert done.downloaded == 15 and done.bytes == 15
    assert sorted(p.name for p in lib.models_dir.iterdir()) == [
        f"m-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
    # Deleting the item removes every part.
    await lib.delete(item.name, lambda n: False)
    assert not list(lib.models_dir.iterdir())


async def test_non_first_split_part_rejected(lib):
    with pytest.raises(LibraryError, match="first part"):
        await lib.add_hf(REPO, "m-00002-of-00003.gguf")
    assert lib.list() == []


@respx.mock
async def test_incomplete_split_rejected(lib):
    respx.get(f"{HF}/api/models/{REPO}/tree/main").respond(json=tree(
        ("file", "m-00001-of-00002.gguf", 4)))
    with pytest.raises(LibraryError, match="incomplete"):
        await lib.add_hf(REPO, "m-00001-of-00002.gguf")
    assert lib.list() == []


@pytest.mark.parametrize("file", ["notgguf.bin", "/abs.gguf", "../x.gguf", "a/../../x.gguf"])
async def test_add_hf_rejects_bad_file(lib, file):
    with pytest.raises(LibraryError):
        await lib.add_hf(REPO, file)


@respx.mock
async def test_name_collision_409(lib):
    respx.get(f"{HF}/{REPO}/resolve/main/m.gguf").respond(content=b"abc")
    await lib.add_hf(REPO, "m.gguf")
    with pytest.raises(LibraryError) as ei:
        await lib.add_hf(REPO, "m.gguf")
    assert ei.value.status == 409
    await wait_status(lib, "m.gguf", "ready")


@respx.mock
async def test_delete_while_downloading_cancels_and_removes_part(lib):
    started = asyncio.Event()

    async def stream():
        yield b"a" * 1000
        started.set()
        await asyncio.sleep(60)
        yield b"b"

    respx.get(f"{HF}/{REPO}/resolve/main/m.gguf").mock(
        return_value=httpx.Response(200, stream=stream()))
    await lib.add_hf(REPO, "m.gguf")
    await asyncio.wait_for(started.wait(), 5)
    task = lib._tasks["m.gguf"]
    await lib.delete("m.gguf", lambda n: False)
    await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled() or task.done()
    assert lib.get("m.gguf") is None
    assert not list(lib.models_dir.iterdir())


@respx.mock
async def test_resume_restarts_downloading_items(tmp_path):
    db = tmp_path / "db.sqlite"
    models = tmp_path / "models"
    models.mkdir()
    lb1 = Library(db, models, http=httpx.AsyncClient())
    # Simulate a crashed process: row says downloading, a stale .part exists, no task runs.
    lb1._exec("INSERT INTO library(name,path,source,hf_repo,hf_file,bytes,downloaded,status,"
              "created_at) VALUES('m.gguf',?, 'hf', ?, 'm.gguf', NULL, 7, 'downloading', 1.0)",
              (str(models / "m.gguf"), REPO))
    (models / "m.gguf.part").write_bytes(b"stale")
    lb1._db.close()

    lb2 = Library(db, models, http=httpx.AsyncClient())
    respx.get(f"{HF}/{REPO}/resolve/main/m.gguf").respond(content=b"fresh-data")
    lb2.resume()
    it = await wait_status(lb2, "m.gguf", "ready")
    assert (models / "m.gguf").read_bytes() == b"fresh-data" and it.downloaded == 10
    assert not list(models.glob("*.part"))
    await lb2.shutdown()


async def test_shutdown_cancels_tasks_and_closes_own_client(tmp_path):
    lb = Library(":memory:", tmp_path)
    with respx.mock:
        async def stream():
            await asyncio.sleep(60)
            yield b"x"

        respx.get(f"{HF}/{REPO}/resolve/main/m.gguf").mock(
            return_value=httpx.Response(200, stream=stream()))
        await lb.add_hf(REPO, "m.gguf")
        await asyncio.sleep(0.05)
        await lb.shutdown()
    assert lb._http.is_closed
    assert not list(tmp_path.glob("*.part"))


# ---- paths -------------------------------------------------------------------------

def test_add_path_ok_and_not_copied(lib, tmp_path):
    f = tmp_path / "ext" / "big.gguf"
    f.parent.mkdir()
    f.write_bytes(b"12345")
    item = lib.add_path(str(f))
    assert (item.name, item.status, item.bytes, item.source) == ("big.gguf", "ready", 5, "path")
    assert not lib.models_dir.exists() or not list(lib.models_dir.iterdir())
    assert lib.resolve("big.gguf") == f


def test_add_path_validations(lib, tmp_path):
    d = tmp_path / "dir.gguf"
    d.mkdir()
    txt = tmp_path / "a.txt"
    txt.write_bytes(b"x")
    for bad in ("rel/x.gguf", str(tmp_path / "missing.gguf"), str(txt), str(d), ""):
        with pytest.raises(LibraryError):
            lib.add_path(bad)
    assert lib.list() == []


def test_add_path_collision(lib, tmp_path):
    a = tmp_path / "a" / "m.gguf"
    b = tmp_path / "b" / "m.gguf"
    for p in (a, b):
        p.parent.mkdir()
        p.write_bytes(b"x")
    lib.add_path(str(a))
    with pytest.raises(LibraryError) as ei:
        lib.add_path(str(b))
    assert ei.value.status == 409


async def test_delete_path_item_keeps_user_file(lib, tmp_path):
    f = tmp_path / "keep.gguf"
    f.write_bytes(b"x")
    lib.add_path(str(f))
    await lib.delete("keep.gguf", lambda n: False)
    assert f.exists() and lib.get("keep.gguf") is None


@respx.mock
async def test_delete_hf_item_removes_file(lib):
    respx.get(f"{HF}/{REPO}/resolve/main/m.gguf").respond(content=b"abc")
    await lib.add_hf(REPO, "m.gguf")
    await wait_status(lib, "m.gguf", "ready")
    await lib.delete("m.gguf", lambda n: False)
    assert not (lib.models_dir / "m.gguf").exists()


async def test_delete_in_use_guard_and_unknown(lib, tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    lib.add_path(str(f))
    with pytest.raises(LibraryError, match="uses this file") as ei:
        await lib.delete("m.gguf", lambda n: n == "m.gguf")
    assert ei.value.status == 409 and lib.get("m.gguf")
    with pytest.raises(LibraryError) as ei:
        await lib.delete("nope.gguf", lambda n: False)
    assert ei.value.status == 404


def test_resolve_requires_ready_and_existing_file(lib, tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    lib.add_path(str(f))
    assert lib.resolve("m.gguf") == f
    f.unlink()
    assert lib.resolve("m.gguf") is None
    assert lib.resolve("../m.gguf") is None
    lib._exec("UPDATE library SET status='failed'")
    f.write_bytes(b"x")
    assert lib.resolve("m.gguf") is None


def _ready_split(lib, total=3):
    lib.models_dir.mkdir(parents=True, exist_ok=True)
    names = [f"m-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]
    for n in names:
        (lib.models_dir / n).write_bytes(n.encode())
    lib._exec(
        "INSERT INTO library(name,path,source,hf_repo,hf_file,bytes,downloaded,status,created_at,parts)"
        " VALUES(?,?,'hf',?,?,1,1,'ready',0,?)",
        (names[0], str(lib.models_dir / names[0]), REPO, names[0], "[]"))
    return names


def test_part_paths_single_and_split(lib, tmp_path):
    f = tmp_path / "s.gguf"
    f.write_bytes(b"x")
    lib.add_path(str(f))
    assert lib.part_paths("s.gguf") == [f]
    assert lib.part_paths("nope.gguf") is None
    names = _ready_split(lib)
    assert lib.part_paths(names[0]) == [lib.models_dir / n for n in names]
    (lib.models_dir / names[1]).unlink()
    assert lib.part_paths(names[0]) is None  # a missing part makes the item unusable
    assert lib.resolve(names[0]) is None


def test_part_paths_none_while_downloading(lib):
    names = _ready_split(lib)
    lib._exec("UPDATE library SET status='downloading'")
    assert lib.part_paths(names[0]) is None
    assert lib.resolve(names[1]) is None


def test_resolve_part_of_ready_split_only(lib):
    names = _ready_split(lib)
    assert lib.resolve(names[1]) == lib.models_dir / names[1]
    assert lib.resolve(names[2]) == lib.models_dir / names[2]
    assert lib.resolve("m-00004-of-00003.gguf") is None  # beyond the part list
    assert lib.resolve("other-00002-of-00003.gguf") is None  # unknown item
    assert lib.resolve("m-00002-of-00005.gguf") is None  # foreign total
    assert lib.resolve("../m-00002-of-00003.gguf") is None
    assert lib.resolve("sub/m-00002-of-00003.gguf") is None


def test_resolve_part_of_path_item_is_not_served(lib, tmp_path):
    # a path item is never split: sibling files next to it are not reachable by part name
    f = tmp_path / "x.gguf"
    f.write_bytes(b"x")
    (tmp_path / "x-00002-of-00003.gguf").write_bytes(b"y")
    lib.add_path(str(f))
    assert lib.resolve("x-00002-of-00003.gguf") is None


def test_add_path_rejects_split_part(lib, tmp_path):
    f = tmp_path / "m-00001-of-00003.gguf"
    f.write_bytes(b"x")
    with pytest.raises(LibraryError, match="split"):
        lib.add_path(str(f))


# ---- HTTP --------------------------------------------------------------------------

@pytest.fixture
def client(lib):
    app = FastAPI()
    used = set()
    app.include_router(make_library_router(lib, require_bearer("admin"), lambda n: n in used))
    app.include_router(make_files_router(lib, require_bearer("cluster")))
    c = TestClient(app)
    c.used = used
    c.lib = lib
    return c


ADMIN = {"Authorization": "Bearer admin"}
CLUSTER = {"Authorization": "Bearer cluster"}


def test_api_auth_required(client):
    assert client.get("/api/library").status_code == 401
    assert client.get("/api/library", headers=CLUSTER).status_code == 401
    assert client.get("/files/x.gguf").status_code == 401


def test_api_add_path_list_delete(client, tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"hello")
    r = client.post("/api/library", json={"path": str(f)}, headers=ADMIN)
    assert r.status_code == 200 and r.json()["name"] == "m.gguf" and r.json()["bytes"] == 5
    assert client.post("/api/library", json={"path": str(f)}, headers=ADMIN).status_code == 409
    assert [i["name"] for i in client.get("/api/library", headers=ADMIN).json()] == ["m.gguf"]
    client.used.add("m.gguf")
    r = client.delete("/api/library/m.gguf", headers=ADMIN)
    assert r.status_code == 409 and "uses this file" in r.json()["detail"]
    client.used.clear()
    assert client.delete("/api/library/m.gguf", headers=ADMIN).status_code == 200
    assert client.delete("/api/library/m.gguf", headers=ADMIN).status_code == 404


def test_api_body_must_have_exactly_one_form(client):
    for body in ({}, {"hf_repo": "a/b"}, {"hf_file": "x.gguf"}, {"path": "/x.gguf", "hf_repo": "a/b",
                                                                  "hf_file": "x.gguf"}):
        assert client.post("/api/library", json=body, headers=ADMIN).status_code == 422
    assert client.post("/api/library", json={"path": "rel.gguf"}, headers=ADMIN).status_code == 400


def test_api_hf_files_maps_errors(client):
    with respx.mock:
        respx.get(f"{HF}/api/models/{REPO}/tree/main").respond(
            json=tree(("file", "a.gguf", 1)))
        r = client.get("/api/hf/files", params={"repo": REPO}, headers=ADMIN)
        assert r.json() == [{"file": "a.gguf", "bytes": 1}]
        respx.get(f"{HF}/api/models/{REPO}/tree/main").respond(404)
        r = client.get("/api/hf/files", params={"repo": REPO}, headers=ADMIN)
        assert r.status_code == 404 and r.json() == {"detail": "repo not found"}
    assert client.get("/api/hf/files", params={"repo": "bad"}, headers=ADMIN).status_code == 400


def test_files_serves_known_rejects_unknown(client, tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"hello")
    client.post("/api/library", json={"path": str(f)}, headers=ADMIN)
    r = client.get("/files/m.gguf", headers=CLUSTER)
    assert r.status_code == 200 and r.content == b"hello"
    assert client.get("/files/unknown.gguf", headers=CLUSTER).status_code == 404
    assert client.get("/files/..%2Fm.gguf", headers=CLUSTER).status_code == 404


def test_files_serves_split_parts(client):
    names = _ready_split(client.lib)
    for n in names:
        r = client.get(f"/files/{n}", headers=CLUSTER)
        assert r.status_code == 200 and r.content == n.encode()
    assert client.get("/files/m-00004-of-00003.gguf", headers=CLUSTER).status_code == 404


# ---- real --------------------------------------------------------------------------

@pytest.mark.real
async def test_real_hf_listing(tmp_path):
    lb = Library(":memory:", tmp_path)
    try:
        files = await lb.hf_files("Qwen/Qwen2.5-0.5B-Instruct-GGUF")
    finally:
        await lb.shutdown()
    assert any(f["file"].endswith("q4_k_m.gguf") and f["bytes"] > 0 for f in files)


@respx.mock
async def test_delete_then_readd_same_name_keeps_new_download_intact(lib):
    # The cancelled task cleans up its .part files; delete must wait for that, or the old task
    # unlinks the NEW download's .part and the new download dies with FileNotFoundError.
    started = asyncio.Event()
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            async def slow():
                yield b"a" * 1000
                started.set()
                await asyncio.sleep(60)
                yield b"b"
            return httpx.Response(200, stream=slow())
        return httpx.Response(200, content=b"new" * 100)

    respx.get(f"{HF}/{REPO}/resolve/main/m.gguf").mock(side_effect=handler)
    await lib.add_hf(REPO, "m.gguf")
    await asyncio.wait_for(started.wait(), 5)
    await lib.delete("m.gguf", lambda n: False)
    assert "m.gguf" not in lib._tasks
    await lib.add_hf(REPO, "m.gguf")
    done = await wait_status(lib, "m.gguf", "ready")
    assert done.downloaded == 300
    assert (lib.models_dir / "m.gguf").read_bytes() == b"new" * 100
    assert not list(lib.models_dir.glob("*.part"))


# ---- host path translation / browsing ----------------------------------------------

def _mapped(tmp_path, **kw):
    return Library(":memory:", tmp_path / "models", http=httpx.AsyncClient(), **kw)


BS = chr(92)


def test_translate_longest_prefix_and_component_boundary(tmp_path):
    lb = _mapped(tmp_path, path_map={"/srv/gguf": "/c/a", "/srv/gguf/big": "/c/b",
                                     "C:" + BS + "models": "/c/win"})
    assert lb.translate_path("/srv/gguf/x.gguf") == str(Path("/c/a/x.gguf"))
    assert lb.translate_path("/srv/gguf/big/y/z.gguf") == str(Path("/c/b/y/z.gguf"))
    assert lb.translate_path("/srv/gguf2/x.gguf") is None  # not a component match
    assert lb.translate_path("C:" + BS + "models" + BS + "sub" + BS + "m.gguf") == str(Path("/c/win/sub/m.gguf"))
    assert lb.translate_path("C:/models//m.gguf") == str(Path("/c/win/m.gguf"))
    assert lb.host_path("/c/b/y/z.gguf") == "/srv/gguf/big/y/z.gguf"
    assert lb.host_path("/c/a") == "/srv/gguf"
    assert lb.host_path("/elsewhere/x.gguf") is None
    lb._db.close()


def test_add_path_translates_host_path_and_stores_container_path(tmp_path):
    cont = tmp_path / "mounted"
    cont.mkdir()
    (cont / "m.gguf").write_bytes(b"abc")
    lb = _mapped(tmp_path, path_map={"/srv/gguf": str(cont)})
    item = lb.add_path("/srv/gguf/m.gguf")
    assert item.path == str(cont / "m.gguf") and item.bytes == 3
    # An existing path is used as given, even when it also matches a mapping.
    lb2 = _mapped(tmp_path, path_map={str(tmp_path): str(cont)})
    assert lb2.add_path(str(cont / "m.gguf")).path == str(cont / "m.gguf")
    lb._db.close()
    lb2._db.close()


def test_add_path_not_found_message_lists_roots_and_map(tmp_path):
    lb = _mapped(tmp_path, path_map={"/srv/gguf": "/models"}, model_roots=["/models"])
    with pytest.raises(LibraryError) as ei:
        lb.add_path("/srv/other/x.gguf")
    msg = ei.value.message
    assert "no such file inside the coordinator: /srv/other/x.gguf" in msg
    assert "/models" in msg and "/models \u2190 /srv/gguf" in msg
    assert "GPUPOOL_HOST_MODELS_DIR" in msg
    lb._db.close()


def _symlink(link, target):
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted here")


def test_add_path_broken_symlink_message(tmp_path):
    link = tmp_path / "m.gguf"
    _symlink(link, tmp_path / "nowhere" / "blob")
    lb = _mapped(tmp_path)
    with pytest.raises(LibraryError) as ei:
        lb.add_path(str(link))
    assert "is a symlink to" in ei.value.message and "not visible inside the coordinator" in ei.value.message
    # Same through a translated path.
    lb2 = _mapped(tmp_path, path_map={"/host": str(tmp_path)})
    with pytest.raises(LibraryError) as ei:
        lb2.add_path("/host/m.gguf")
    assert "is a symlink to" in ei.value.message
    lb._db.close()
    lb2._db.close()


def test_browse_flags_depth_and_sorting(tmp_path):
    root = tmp_path / "roots"
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "one.gguf").write_bytes(b"1")
    (root / "b.gguf").write_bytes(b"22")
    (root / "notes.txt").write_bytes(b"x")
    (root / "m-00001-of-00002.gguf").write_bytes(b"3")
    (root / "m-00002-of-00002.gguf").write_bytes(b"3")
    deep = root
    for i in range(8):
        deep = deep / f"d{i}"
    deep.mkdir(parents=True)
    (deep / "toodeep.gguf").write_bytes(b"x")
    (root / "d0" / "d1" / "d2" / "d3" / "d4" / "d5").mkdir(exist_ok=True)
    (root / "d0" / "d1" / "d2" / "d3" / "d4" / "d5" / "ok.gguf").write_bytes(b"x")
    lb = _mapped(tmp_path, model_roots=[str(root), str(tmp_path / "gone")],
                 path_map={"/srv": str(root)})
    lb.add_path(str(root / "b.gguf"))
    out = lb.browse()
    names = [f["name"] for f in out["files"]]
    assert names == sorted(names, key=lambda n: next(f["path"] for f in out["files"] if f["name"] == n))
    assert "notes.txt" not in names and "toodeep.gguf" not in names and "ok.gguf" in names
    by = {f["name"]: f for f in out["files"]}
    assert by["b.gguf"]["in_library"] and not by["one.gguf"]["in_library"]
    assert by["b.gguf"]["bytes"] == 2 and by["b.gguf"]["host_path"] == "/srv/b.gguf"
    assert by["m-00002-of-00002.gguf"]["split_part"] and not by["b.gguf"]["split_part"]
    assert [r["exists"] for r in out["roots"]] == [True, False]
    assert out["roots"][0]["host_path"] == "/srv" and out["roots"][1]["host_path"] is None
    assert out["truncated"] is False
    lb._db.close()


def test_browse_defaults_to_models_dir_and_truncates(tmp_path):
    lb = _mapped(tmp_path)
    (tmp_path / "models").mkdir()
    for i in range(5):
        (tmp_path / "models" / f"f{i}.gguf").write_bytes(b"x")
    out = lb.browse(max_files=3)
    assert [r["path"] for r in out["roots"]] == [str(tmp_path / "models")]
    assert len(out["files"]) == 3 and out["truncated"] is True
    assert lb.browse(max_files=5)["truncated"] is False
    lb._db.close()


def test_browse_broken_link_and_dir_symlink_loop(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    (root / "real.gguf").write_bytes(b"x")
    _symlink(root / "dead.gguf", root / "missing")
    _symlink(root / "loop", root)  # directory symlink back to the root: must not recurse
    lb = _mapped(tmp_path, model_roots=[str(root)])
    out = lb.browse()
    by = {f["name"]: f for f in out["files"]}
    assert set(by) == {"real.gguf", "dead.gguf"}
    assert by["dead.gguf"]["broken_link"] and by["dead.gguf"]["bytes"] == 0
    assert not by["real.gguf"]["broken_link"]
    lb._db.close()


def test_api_browse(client, tmp_path):
    (client.lib.models_dir).mkdir()
    (client.lib.models_dir / "a.gguf").write_bytes(b"x")
    assert client.get("/api/library/browse").status_code == 401
    r = client.get("/api/library/browse", headers=ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert [f["name"] for f in body["files"]] == ["a.gguf"] and body["truncated"] is False
    assert body["roots"][0]["exists"] is True

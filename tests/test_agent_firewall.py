from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from gpupool.agent import procs
from gpupool.agent.firewall import CHAIN, RpcFirewall
from gpupool.agent.procs import ProcessManager
from gpupool.common.models import EngineSpec

BINS = {"server": Path("llama-server"), "rpc": Path("ggml-rpc-server")}


class FakeIpt:
    """Records every call; emulates chains just enough for -N/-L/-F/-C/-I/-A/-D."""

    def __init__(self, missing=(), deny=False):
        self.calls: list[list[str]] = []
        self.missing, self.deny = set(missing), deny
        self.chain = {"iptables": [], "ip6tables": []}
        self.created = {"iptables": False, "ip6tables": False}
        self.hooked = {"iptables": False, "ip6tables": False}

    def __call__(self, argv):
        self.calls.append(argv)
        b, args = argv[0], argv[2:]  # argv[1] is -w
        if b in self.missing:
            return 127, f"{b}: not found"
        if self.deny:
            return 4, "Permission denied (you must be root)"
        op = args[0]
        if op == "-N":
            if self.created[b]:
                return 1, "Chain already exists"
            self.created[b] = True
        elif op == "-L":
            return (0, "") if self.created[b] else (1, "No chain")
        elif op == "-F":
            self.chain[b].clear()
        elif op == "-C":
            return (0, "") if self.hooked[b] else (1, "no match")
        elif op == "-I":
            self.hooked[b] = True
        elif op == "-A":
            self.chain[b].append(tuple(args[2:]))  # args[1] is the chain name
        elif op == "-D":
            try:
                self.chain[b].remove(tuple(args[2:]))
            except ValueError:
                return 1, "Bad rule"
        return 0, ""


def resolver(host):
    return {"head": ["10.0.0.1"], "dual": ["10.0.0.2", "fd00::2"]}.get(host, [])


def fw(**kw):
    ipt = FakeIpt(**kw)
    f = RpcFirewall(runner=ipt, resolver=resolver)
    f.setup()
    return f, ipt


def sources(rules):
    """Source addresses of the RETURN rules (index 5 of ('-p','tcp','--dport',P,'-s',IP,...))."""
    return [r[5] for r in rules if "RETURN" in r]


def test_setup_creates_and_hooks_chain_once_and_is_idempotent():
    f, ipt = fw()
    assert f.active
    f.setup()
    hooks = [c for c in ipt.calls if c[2] == "-I" and c[0] == "iptables"]
    assert len(hooks) == 1 and hooks[0][-2:] == ["-j", CHAIN]


def test_setup_flushes_stale_rules():
    f, ipt = fw()
    f.install(9001, ["10.0.0.1"])
    assert ipt.chain["iptables"]
    f.setup()
    assert ipt.chain["iptables"] == []


def test_install_allows_peers_and_loopback_then_drops():
    f, ipt = fw()
    h = f.install(9001, ["10.0.0.1", "head"])
    rules = ipt.chain["iptables"]
    assert sources(rules) == ["10.0.0.1", "127.0.0.1"]  # "head" resolves to the same IP
    assert rules[-1] == ("-p", "tcp", "--dport", "9001", "-j", "DROP")  # catch-all comes last
    # IPv6 gets loopback + drop, so it is not an open back door
    assert sources(ipt.chain["ip6tables"]) == ["::1"]
    assert ipt.chain["ip6tables"][-1][-1] == "DROP"
    f.remove(h)
    assert ipt.chain["iptables"] == [] and ipt.chain["ip6tables"] == []


def test_hostname_resolves_to_both_families():
    f, ipt = fw()
    f.install(9002, ["dual"])
    assert "10.0.0.2" in sources(ipt.chain["iptables"])
    assert "fd00::2" in sources(ipt.chain["ip6tables"])


def test_unresolvable_peer_is_refused_and_logged(caplog):
    f, ipt = fw()
    with caplog.at_level(logging.ERROR):
        f.install(9003, ["nosuchhost"])
    assert sources(ipt.chain["iptables"]) == ["127.0.0.1"]
    assert "cannot resolve" in caplog.text


def test_remove_only_its_own_rules():
    f, ipt = fw()
    a = f.install(9001, ["10.0.0.1"])
    f.install(9002, ["10.0.0.1"])
    f.remove(a)
    assert {r[3] for r in ipt.chain["iptables"]} == {"9002"}


@pytest.mark.parametrize("kw", [{"missing": ["iptables", "ip6tables"]}, {"deny": True}])
def test_missing_or_forbidden_iptables_is_inactive_with_one_error(kw, caplog):
    with caplog.at_level(logging.ERROR):
        f, ipt = fw(**kw)
        assert not f.active
        assert f.install(9001, ["10.0.0.1"]) == []
    assert len([r for r in caplog.records if r.levelno == logging.ERROR]) == 1
    assert not any(c[2] == "-A" for c in ipt.calls)


def test_no_ip6tables_keeps_v4_active(caplog):
    with caplog.at_level(logging.WARNING):
        f, ipt = fw(missing=["ip6tables"])
    assert f.active
    f.install(9001, ["10.0.0.1"])
    assert ipt.chain["iptables"] and not ipt.chain["ip6tables"]
    assert "ip6tables unavailable" in caplog.text


def test_failed_rule_rolls_back_and_leaves_port_open():
    f, ipt = fw()
    n = {"a": 0}

    def flaky(argv):
        if argv[2] == "-A":
            n["a"] += 1
            if n["a"] == 2:
                return 1, "boom"
        return ipt(argv)

    f._run = flaky
    assert f.install(9001, ["10.0.0.1"]) == []
    assert ipt.chain["iptables"] == []  # no half-installed allow rules left behind


# ---------- ProcessManager integration ----------

SLEEPER = "import time; time.sleep(60)"


@pytest.fixture
def pm_fw(tmp_path, monkeypatch):
    monkeypatch.setattr(procs, "build_command",
                        lambda spec, bins, host, mp: [sys.executable, "-c", SLEEPER])
    monkeypatch.setattr(ProcessManager, "_binaries", lambda self: BINS)
    f, ipt = fw()
    m = ProcessManager(tmp_path, tmp_path / "logs", "127.0.0.1", rpc_firewall=True, firewall=f)
    yield m, ipt
    m.stop_all()


def spec(eid="e1", port=19871, kind="rpc", peers=("10.0.0.1",)):
    return EngineSpec(engine_id=eid, kind=kind, port=port, devices=["CPU"],
                      allowed_peers=list(peers), model="m")


def test_rpc_engine_rules_installed_on_start_and_removed_on_stop(pm_fw):
    pm, ipt = pm_fw
    pm.start(spec())
    assert ipt.chain["iptables"][-1][-1] == "DROP"
    pm.stop("e1")
    assert ipt.chain["iptables"] == []


def test_own_bind_address_is_allowed(tmp_path, monkeypatch):
    # Real-cluster regression: RPC bound to the container IP; the agent's readiness probe to
    # its own IP has that IP as source, was dropped, and the engine never looked "running".
    monkeypatch.setattr(procs, "build_command",
                        lambda spec, bins, host, mp: [sys.executable, "-c", SLEEPER])
    monkeypatch.setattr(ProcessManager, "_binaries", lambda self: BINS)
    monkeypatch.setattr(ProcessManager, "_port_free", lambda self, port: True)  # no such IP here
    f, ipt = fw()
    m = ProcessManager(tmp_path, tmp_path / "logs", "172.30.0.13", rpc_firewall=True, firewall=f)
    try:
        m.start(spec(peers=("172.30.0.11",)))
        sources = [r[r.index("-s") + 1] for r in ipt.chain["iptables"] if "-s" in r]
        assert "172.30.0.11/32" in sources or "172.30.0.11" in sources
        assert "172.30.0.13/32" in sources or "172.30.0.13" in sources
    finally:
        m.stop_all()


def test_stop_all_removes_rules(pm_fw):
    pm, ipt = pm_fw
    pm.start(spec("a", 19871))
    pm.start(spec("b", 19872))
    assert len(ipt.chain["iptables"]) == 6
    pm.stop_all()
    assert ipt.chain["iptables"] == []


def test_restarting_a_dead_engine_does_not_leak_or_double_rules(pm_fw):
    pm, ipt = pm_fw
    pm.start(spec())
    pm._engines["e1"].proc.kill()
    pm._engines["e1"].proc.wait()
    n = len(ipt.chain["iptables"])
    pm.start(spec())
    assert len(ipt.chain["iptables"]) == n
    pm.stop("e1")
    assert ipt.chain["iptables"] == []


def test_no_rules_without_peers_or_for_server_engines(pm_fw):
    pm, ipt = pm_fw
    pm.start(spec("a", 19871, peers=()))
    pm.start(spec("b", 19872, kind="server"))
    assert ipt.chain["iptables"] == []


def test_no_firewall_when_off(tmp_path, monkeypatch):
    ipt = FakeIpt()
    monkeypatch.setattr(procs, "RpcFirewall", lambda: RpcFirewall(runner=ipt, resolver=resolver))
    m = ProcessManager(tmp_path, tmp_path / "logs", "127.0.0.1")
    assert m.firewall is None and ipt.calls == []


@pytest.mark.parametrize("host,warns", [("8.8.8.8", True), ("0.0.0.0", True), ("10.1.2.3", False),
                                        ("127.0.0.1", False), ("169.254.1.1", False),
                                        ("192.168.0.9", False)])
def test_warns_on_public_bind_without_firewall(tmp_path, caplog, host, warns):
    with caplog.at_level(logging.WARNING, logger="gpupool.agent.procs"):
        ProcessManager(tmp_path, tmp_path / "logs", host)
    assert ("GPUPOOL_RPC_FIREWALL" in caplog.text) == warns


def test_no_warning_when_firewall_on(tmp_path, caplog):
    f, _ = fw()
    with caplog.at_level(logging.WARNING, logger="gpupool.agent.procs"):
        ProcessManager(tmp_path, tmp_path / "logs", "8.8.8.8", rpc_firewall=True, firewall=f)
    assert "GPUPOOL_RPC_FIREWALL" not in caplog.text

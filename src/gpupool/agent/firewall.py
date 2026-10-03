"""Restrict ggml-rpc-server ports to their head node with iptables.

ggml-rpc-server has no authentication: whoever reaches the port can allocate GPU memory, run
graphs and read/write tensors. All rules live in one dedicated chain (jumped to from INPUT once),
so a restart can flush everything this agent ever installed without touching other firewall rules.
"""
from __future__ import annotations

import ipaddress
import logging
import socket
import subprocess
import threading
from typing import Callable

log = logging.getLogger(__name__)

CHAIN = "GPUPOOL-RPC"
# (iptables binary, rule args after the chain name); also the exact spec used to delete it.
Rule = tuple[str, tuple[str, ...]]
# runner(argv) -> (returncode, combined output). 127 = binary not found.
Runner = Callable[[list[str]], tuple[int, str]]
Resolver = Callable[[str], list[str]]


def default_runner(argv: list[str]) -> tuple[int, str]:
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=15, errors="replace")
    except FileNotFoundError:
        return 127, f"{argv[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{argv[0]}: timed out"
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def default_resolver(host: str) -> list[str]:
    try:
        return sorted({i[4][0] for i in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)})
    except socket.gaierror:
        return []


class RpcFirewall:
    def __init__(self, runner: Runner = default_runner, resolver: Resolver = default_resolver):
        self._run = runner
        self._resolve = resolver
        self._lock = threading.Lock()
        self.active = False  # False = rules cannot be installed; engines run unprotected
        self._v6 = False

    def _ipt(self, binary: str, *args: str) -> tuple[int, str]:
        return self._run([binary, "-w", *args])  # -w: wait for the xtables lock, don't fail

    def _setup_one(self, binary: str) -> str | None:
        """Create/flush the chain and hook it into INPUT. Returns an error text or None."""
        rc, out = self._ipt(binary, "-N", CHAIN)
        if rc != 0 and self._ipt(binary, "-L", CHAIN, "-n")[0] != 0:
            return out.strip() or f"{binary} exited {rc}"  # missing, or no NET_ADMIN/root
        # Engines of a previous agent were reaped, so every rule in the chain is stale.
        rc, out = self._ipt(binary, "-F", CHAIN)
        if rc != 0:
            return out.strip()
        if self._ipt(binary, "-C", "INPUT", "-j", CHAIN)[0] != 0:
            rc, out = self._ipt(binary, "-I", "INPUT", "1", "-j", CHAIN)
            if rc != 0:
                return out.strip()
        return None

    def setup(self) -> bool:
        """Idempotent. Sets and returns `active`; never raises."""
        with self._lock:
            err = self._setup_one("iptables")
            if err is not None:
                log.error("RPC firewall unavailable (%s); RPC ports are NOT restricted. It needs "
                          "root and iptables (Docker: --cap-add NET_ADMIN). Protect ports "
                          "9000-9999 with your own firewall.", err)
                self.active = False
                return False
            err6 = self._setup_one("ip6tables")
            self._v6 = err6 is None
            if not self._v6:
                log.warning("ip6tables unavailable (%s): IPv6 access to RPC ports is not "
                            "restricted", err6)
            self.active = True
            return True

    def _rules_for(self, port: int, peers: list[str]) -> list[Rule]:
        v4 = {"127.0.0.1"}  # the agent's own readiness probe
        v6 = {"::1"}
        for p in peers:
            try:
                addrs = [str(ipaddress.ip_address(p))]
            except ValueError:
                addrs = self._resolve(p)
                if not addrs:
                    log.error("RPC firewall: cannot resolve peer %r; it will be refused", p)
            for a in addrs:
                try:
                    ip = ipaddress.ip_address(a.split("%")[0])
                except ValueError:
                    continue
                (v4 if ip.version == 4 else v6).add(str(ip))
        rules: list[Rule] = []
        for binary, srcs in (("iptables", v4), ("ip6tables", v6)):
            if binary == "ip6tables" and not self._v6:
                continue
            base = ("-p", "tcp", "--dport", str(port))
            # allows first, the catch-all DROP last; the chain is appended to, never reordered
            rules += [(binary, (*base, "-s", s, "-j", "RETURN")) for s in sorted(srcs)]
            rules.append((binary, (*base, "-j", "DROP")))
        return rules

    def install(self, port: int, peers: list[str]) -> list[Rule]:
        """Allow only `peers` (and loopback) to reach `port`. Returns a handle for remove().
        Empty handle = nothing installed (inactive, or a rule failed and was rolled back)."""
        if not self.active:
            return []
        done: list[Rule] = []
        with self._lock:
            for binary, args in self._rules_for(port, peers):
                rc, out = self._ipt(binary, "-A", CHAIN, *args)
                if rc != 0:
                    log.error("RPC firewall: %s rule for port %d failed (%s); port left "
                              "unrestricted", binary, port, out.strip())
                    self._remove_locked(done)
                    return []
                done.append((binary, args))
        return done

    def remove(self, handle: list[Rule]) -> None:
        with self._lock:
            self._remove_locked(handle)

    def _remove_locked(self, handle: list[Rule]) -> None:
        for binary, args in reversed(handle):
            rc, out = self._ipt(binary, "-D", CHAIN, *args)
            if rc != 0:
                log.warning("RPC firewall: could not delete rule %s %s (%s)", binary,
                            " ".join(args), out.strip())

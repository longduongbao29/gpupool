"""Static checks for the CI end-to-end setup (the real run is scripts/ci_e2e.py)."""
from pathlib import Path

import pytest

# PyYAML is only a transitive dependency: skip rather than fail if it ever disappears.
yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]


def test_ci_compose_is_cpu_only_and_splits_the_model() -> None:
    doc = yaml.safe_load((ROOT / "docker-compose.ci.yml").read_text())
    svc = doc["services"]
    assert set(svc) == {"coordinator", "server-a", "server-b", "server-c"}
    for name in ("server-a", "server-b", "server-c"):
        s = svc[name]
        assert "deploy" not in s, "CI hosts have no GPU"
        env = s["environment"]
        assert env["GPUPOOL_INCLUDE_CPU"] == "true"
        assert "CPU" in env["GPUPOOL_BUDGET_MB"]
        assert env["GPUPOOL_JOIN"].startswith("http://172.31.77.10:8080#")


def test_ci_budget_is_smaller_than_the_model() -> None:
    import re
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import ci_e2e

    compose = (ROOT / "docker-compose.ci.yml").read_text()
    budget = int(re.search(r"CI_BUDGET_MB:-(\d+)", compose).group(1))
    assert budget * 2**20 < ci_e2e.MODEL_BYTES, "the model must not fit one server"
    assert len(ci_e2e.MODEL_SHA256) == 64

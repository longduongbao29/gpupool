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


def test_ci_e2e_has_a_skippable_conversion_stage() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import ci_e2e

    src = (ROOT / "scripts" / "ci_e2e.py").read_text(encoding="utf-8")
    assert "--skip-convert" in src and callable(ci_e2e.run_convert_checks)
    assert ci_e2e.CONVERT_REPO == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert ci_e2e.CONVERT_QUANT == "Q4_K_M"
    # The stage talks to the API contract of the conversion job system.
    for route in ("/api/convert/options", "/api/convert", "/api/library"):
        assert route in src
    assert {"failed", "needs_review", "cancelled"} <= set(ci_e2e.CONVERT_FAIL_STATES)


def test_ci_coordinator_can_reach_huggingface() -> None:
    doc = yaml.safe_load((ROOT / "docker-compose.ci.yml").read_text())
    assert "HF_TOKEN" in doc["services"]["coordinator"]["environment"]


def test_coordinator_dockerfile_selects_the_toolchain_with_a_build_arg() -> None:
    text = (ROOT / "docker" / "coordinator.Dockerfile").read_text(encoding="utf-8")
    assert "ARG WITH_CONVERT=1" in text
    # Stage selection (not a conditional RUN), so BuildKit skips the toolchain when WITH_CONVERT=0.
    assert "FROM final-${WITH_CONVERT}" in text
    assert "AS final-0" in text and "AS final-1" in text
    assert "TORCH_INDEX_URL" in text
    for target in ("llama-quantize", "llama-tokenize", "llama-simple"):
        assert target in text
    # The three paths the coordinator's config reads must be set together, only in final-1.
    tail = text.split("AS final-1", 1)[1]
    for var in ("GPUPOOL_CONVERT_DIR=/opt/llama.cpp", "GPUPOOL_CONVERT_PYTHON=/opt/convert-venv/bin/python",
                "GPUPOOL_LLAMA_TOOLS_DIR=/opt/llama/bin"):
        assert var in tail
        assert var not in text.split("AS final-1", 1)[0]


def test_coordinator_compose_passes_convert_build_args() -> None:
    doc = yaml.safe_load((ROOT / "docker-compose.coordinator.yml").read_text())
    args = doc["services"]["coordinator"]["build"]["args"]
    assert args["WITH_CONVERT"] == "${WITH_CONVERT:-1}"
    assert args["TORCH_INDEX_URL"].startswith("${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu")


def test_workflow_builds_and_smoke_tests_the_toolchain_image() -> None:
    text = (ROOT / ".github" / "workflows" / "docker.yml").read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    jobs = doc["jobs"]

    def build_args(job: str, dockerfile: str) -> str:
        step = next(s for s in jobs[job]["steps"] if s.get("with", {}).get("file") == dockerfile)
        return step["with"].get("build-args", "")

    # Published image and the e2e image share the build args, otherwise the gha cache would miss.
    assert "WITH_CONVERT=1" in build_args("coordinator", "docker/coordinator.Dockerfile")
    assert "WITH_CONVERT=1" in build_args("e2e", "docker/coordinator.Dockerfile")
    assert "--print-supported-models" in text and "llama-quantize" in text


def test_ci_e2e_covers_folder_source_and_imatrix() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import ci_e2e

    src = (ROOT / "scripts" / "ci_e2e.py").read_text(encoding="utf-8")
    # The folder is downloaded with the converter's own file selection (no *.py, no pickles).
    assert set(ci_e2e.FOLDER_FILES) == {
        "config.json", "generation_config.json", "merges.txt", "model.safetensors",
        "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"}
    assert not any(n.endswith(".py") for n in ci_e2e.FOLDER_FILES)
    # Direct converter type: no llama-quantize step, so no imatrix either.
    assert ci_e2e.FOLDER_QUANT == "Q8_0"
    assert ci_e2e.IMATRIX_QUANT == "IQ2_XS" and 0 < ci_e2e.IMATRIX_CHUNKS <= 8
    # Distinct output names, so no job collides with another or with the first conversion.
    names = {ci_e2e.FOLDER_OUT_NAME, ci_e2e.IMATRIX_OUT_NAME}
    assert len(names) == 2 and all(n.endswith(".gguf") for n in names)
    assert "imatrix_chunks" in src and "imatrix_used" in src and "imatrix_available" in src
    # The folder is created under the models dir the compose file mounts at /models, which the
    # coordinator image lists in GPUPOOL_MODEL_ROOTS.
    compose = yaml.safe_load((ROOT / "docker-compose.ci.yml").read_text())
    assert any(v.endswith(":/models:ro") for v in compose["services"]["coordinator"]["volumes"])
    assert "/models/{FOLDER_SUBDIR}" in src
    assert "GPUPOOL_MODEL_ROOTS=/models" in (ROOT / "docker" / "coordinator.Dockerfile").read_text(encoding="utf-8")


def test_skip_convert_skips_every_conversion_stage() -> None:
    import ast

    tree = ast.parse((ROOT / "scripts" / "ci_e2e.py").read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    guarded = [n for n in ast.walk(main) if isinstance(n, ast.If) and "skip_convert" in ast.unparse(n.test)]
    assert guarded, "main() must branch on --skip-convert"
    body = "\n".join(ast.unparse(s) for g in guarded for s in g.body)
    # Folder and imatrix stages run inside run_convert_checks, behind the same switch.
    assert "run_convert_checks" in body
    assert "ensure_folder_source" not in ast.unparse(main)


def test_coordinator_image_builds_and_smoke_tests_llama_imatrix() -> None:
    text = (ROOT / "docker" / "coordinator.Dockerfile").read_text(encoding="utf-8")
    build = text.split("--target", 1)[1].split("&&", 1)[0]
    assert "llama-imatrix" in build
    assert "build/bin/llama-imatrix" in text
    wf = (ROOT / ".github" / "workflows" / "docker.yml").read_text(encoding="utf-8")
    assert "llama-imatrix" in wf and "--help" in wf.split("llama-imatrix", 1)[1].split("\n", 1)[0]

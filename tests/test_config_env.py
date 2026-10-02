from pathlib import Path

from gpupool.common.config import AgentConfig, CoordinatorConfig, env_overrides


def test_env_overrides_types():
    env = {"GPUPOOL_API_KEYS": "a, b,,c", "GPUPOOL_PORT_RANGE": "9100-9200",
           "GPUPOOL_PORT": "8081", "HF_TOKEN": "hf_x", "GPUPOOL_MODELS_DIR": "/data/models"}
    cfg = CoordinatorConfig(**env_overrides(CoordinatorConfig, env))
    assert cfg.api_keys == ["a", "b", "c"]
    assert cfg.port_range == (9100, 9200) and cfg.port == 8081
    assert cfg.hf_token == "hf_x" and cfg.models_dir == Path("/data/models")
    assert CoordinatorConfig().max_request_mb == 32
    assert CoordinatorConfig(**env_overrides(
        CoordinatorConfig, {"GPUPOOL_MAX_REQUEST_MB": "5"})).max_request_mb == 5


def test_env_overrides_agent_dict_and_bool():
    env = {"GPUPOOL_NODE_ID": "a", "GPUPOOL_LLAMA_DIR": "/opt/llama",
           "GPUPOOL_BUDGET_MB": '{"CUDA0": 1000}', "GPUPOOL_INCLUDE_CPU": "true"}
    cfg = AgentConfig(**env_overrides(AgentConfig, env))
    assert cfg.budget_mb == {"CUDA0": 1000} and cfg.include_cpu is True

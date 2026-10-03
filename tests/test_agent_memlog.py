from gpupool.agent.memlog import parse_buffers

REAL = """\
0.00.847.107 I load_tensors:   CPU_Mapped model buffer size =    89.26 MiB
0.00.847.109 I load_tensors:        CUDA0 model buffer size =   373.73 MiB
0.01.024.013 I llama_context:  CUDA_Host  output buffer size =     2.32 MiB
0.01.024.240 I llama_kv_cache:      CUDA0 KV buffer size =    24.00 MiB
0.01.032.483 I sched_reserve:      CUDA0 compute buffer size =    35.76 MiB
0.01.032.487 I sched_reserve:  CUDA_Host compute buffer size =    5.51 MiB
"""

# observed with llama.cpp b11342, one local GPU + one rpc-server
RPC = """\
0.01.162.451 I load_tensors:   CPU_Mapped model buffer size =    89.26 MiB
0.01.162.452 I load_tensors:        CUDA0 model buffer size =   129.39 MiB
0.01.162.453 I load_tensors: RPC0[127.0.0.1:9555] model buffer size =   244.33 MiB
0.01.842.237 I llama_context:        CPU  output buffer size =     2.32 MiB
0.01.842.510 I llama_kv_cache:      CUDA0 KV buffer size =    13.00 MiB
0.01.844.260 I llama_kv_cache: RPC0[127.0.0.1:9555] KV buffer size =    11.00 MiB
0.01.855.957 I sched_reserve:      CUDA0 compute buffer size =    48.79 MiB
0.01.855.964 I sched_reserve: RPC0[127.0.0.1:9555] compute buffer size =    48.79 MiB
0.01.855.965 I sched_reserve:  CUDA_Host compute buffer size =    16.80 MiB
"""


def test_real_lines_host_buffers_excluded():
    assert parse_buffers(REAL) == {
        "CUDA0": {"model_mb": 373.73, "kv_mb": 24.0, "compute_mb": 35.76, "total_mb": 433.49}}


def test_rpc_devices_normalised():
    r = parse_buffers(RPC)
    assert set(r) == {"CUDA0", "RPC0"}
    assert r["RPC0"] == {"model_mb": 244.33, "kv_mb": 11.0, "compute_mb": 48.79, "total_mb": 304.12}
    assert r["CUDA0"]["total_mb"] == 191.18


def test_last_occurrence_wins():
    t = ("I sched_reserve: CUDA0 compute buffer size = 10.00 MiB\n"
         "I sched_reserve: CUDA0 compute buffer size = 35.76 MiB\n")
    assert parse_buffers(t)["CUDA0"]["compute_mb"] == 35.76


def test_missing_kinds_are_zero_and_garbage_ignored():
    assert parse_buffers("") == {}
    assert parse_buffers("hello\nbuffer size = x MiB\nCUDA0 model buffer size = abc MiB\n") == {}
    r = parse_buffers("load_tensors: CUDA1 model buffer size = 5.50 MiB\n")
    assert r == {"CUDA1": {"model_mb": 5.5, "kv_mb": 0.0, "compute_mb": 0.0, "total_mb": 5.5}}

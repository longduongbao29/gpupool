# gpupool — Quick start

> English version. Vietnamese version: [QUICKSTART.vi.md](QUICKSTART.vi.md). Keep both in sync.

Three steps, one command each. Nothing to configure by hand: keys are generated, servers find their
own name and IP, and they join the pool by themselves.

```
1. coordinator machine   docker run ... gpupool-coordinator        (once)
2. every GPU server      docker run ... gpupool-agent  (join cmd)   (once per server)
3. your application     base_url = http://<coordinator>:8080/v1    (any OpenAI client)
```

## 1. Start the coordinator (one machine, no GPU needed)

```bash
docker run -d --name gpupool --restart unless-stopped -p 8080:8080 -v gpupool:/data \
  ghcr.io/longduongbao29/gpupool-coordinator
docker logs gpupool
```

The log prints the **admin key**. Open `http://<this machine's IP>:8080` in a browser and log in with it.
Keys are generated on first start and stored in the `gpupool` volume, so they stay the same after a restart.

The log also prints a join command, but inside Docker it shows the container's internal IP
(`172.17.x.x`). Take the join command from the UI instead: it uses the address you opened the UI with.

## 2. Add each GPU server (one command per server)

In the UI open **Servers → Add Server**, copy the join command and run it on the GPU server.
It looks like this:

```bash
docker run -d --name gpupool-agent --restart unless-stopped --gpus all --network host --pid host \
  -v gpupool-agent:/data \
  -e GPUPOOL_JOIN="http://10.0.0.1:8080#<cluster-token>" \
  ghcr.io/longduongbao29/gpupool-agent
```

The server appears in the UI within a few seconds with all its GPUs. Its name is the hostname and its
IP is detected automatically.

Requirements on the GPU server: an NVIDIA driver ≥ 525, Docker and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Check with:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 3. Serve a model (in the UI)

1. **Models → Add model**: type a Hugging Face repo (e.g. `Qwen/Qwen2.5-7B-Instruct-GGUF`), pick a
   `.gguf` file, **Download**. Or the **Path** tab for a file already on the coordinator machine.
2. **New model**: give it a name (this is the `model` your clients will use), pick the file.
   GPUs: leave **Auto**, or tick the GPUs to use.
3. **Start**. The status goes *starting → running*. **Stop** frees the GPUs.

If the model is bigger than any single GPU, it is split across GPUs and servers automatically.

## 4. Connect your application

Any OpenAI-compatible client works. Only two values change:

| Setting | Value |
| --- | --- |
| Base URL | `http://<coordinator>:8080/v1` |
| Model | the name you gave the model in the UI |
| API key | none needed by default; if you set `GPUPOOL_API_KEYS`, one of those keys |

curl:

```bash
curl http://10.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen7b", "messages": [{"role": "user", "content": "Hello"}]}'
```

Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://10.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(model="qwen7b", messages=[{"role": "user", "content": "Hello"}])
print(r.choices[0].message.content)
```

Tools like Open WebUI, LangChain, LlamaIndex or Continue: choose "OpenAI-compatible" and enter the same
base URL. Streaming (`stream: true`) is supported. The Settings page of the UI shows these snippets with
your real address.

To require a key from clients, start the coordinator with `-e GPUPOOL_API_KEYS=key1,key2`.

## Without Docker

Coordinator:

```bash
git clone https://github.com/longduongbao29/multi-gpu-inference && cd multi-gpu-inference
uv sync && uv run gpupool coordinator          # prints the same admin key and join command
```

GPU server (needs [uv](https://docs.astral.sh/uv/) and a llama.cpp build with CUDA and RPC, b11342):

```bash
uv run gpupool agent --join "http://10.0.0.1:8080#<cluster-token>" --llama-dir /path/to/llama.cpp/build/bin
```

## Network

| Port | Open between | Used for |
| --- | --- | --- |
| 8080 | clients and GPU servers → coordinator | UI, API, model downloads |
| 7070 | coordinator → GPU servers | starting and stopping engines |
| 9000–9999 | coordinator and GPU servers → GPU servers | llama.cpp (HTTP and RPC between servers) |

llama.cpp RPC is not encrypted: keep the GPU servers on a trusted internal network.

## Optional settings (environment variables on the coordinator)

| Variable | Default | Purpose |
| --- | --- | --- |
| `GPUPOOL_API_KEYS` | empty (open) | comma-separated keys clients must send |
| `GPUPOOL_PUBLIC_URL` | detected | address servers use to reach the coordinator, if detection is wrong |
| `GPUPOOL_WEBHOOK_URL` | empty | Slack/Discord webhook for alerts (server down, GPU lost, not enough VRAM) |
| `HF_TOKEN` | empty | for gated or private Hugging Face repos |
| `GPUPOOL_ADMIN_KEY`, `GPUPOOL_CLUSTER_TOKEN` | generated | set your own instead of generated ones |

On a GPU server: `GPUPOOL_MARGIN_PCT` (default `0.10`) is the share of each GPU's memory always left
free for other users.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| Server never appears in the UI | `docker logs gpupool-agent`: "connection refused" → port 8080 blocked or wrong address in the join command (use the coordinator's LAN IP); "wrong cluster token" → copy the join command again |
| Agent log says the server was removed | it was deleted in the UI; add it again there (Servers → Add Server → agent URL `http://<server-ip>:7070`) |
| Model stuck in *failed*: "not enough VRAM" | free GPUs, enable more GPUs, add a server, or use a smaller quantization |
| `could not select device driver "" with capabilities: [[gpu]]` | the NVIDIA Container Toolkit is not installed on that server |

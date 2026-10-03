# Security policy

> English. Tiếng Việt: [SECURITY.vi.md](SECURITY.vi.md)

## Supported versions

| Version | Supported |
| --- | --- |
| 0.3.x | yes |
| older | no |

Fixes go into the latest 0.3.x release; upgrade to receive them.

## Reporting a vulnerability

**Do not open a public issue.** Report privately through GitHub:

1. Open <https://github.com/longduongbao29/gpupool/security/advisories/new> (repository, *Security* tab,
   *Report a vulnerability*).
2. Describe the problem as below. Only the maintainer can see the report.

Please include:

- the gpupool version or image tag, and how it is deployed (Docker or native, coordinator and agent);
- what you found and its impact (what an attacker can read, change or run, and from where on the network);
- steps to reproduce, ideally minimal, with relevant configuration (never include real secrets, tokens or keys);
- logs or a proof of concept if you have one, and any fix you suggest.

This is a small project maintained by one person. We aim to acknowledge reports and fix confirmed issues on a best
effort basis; there is **no guaranteed response or fix time**. Please give us reasonable time to fix before you
disclose publicly; we will credit you in the advisory if you wish.

## Security model

gpupool is meant for a private network or VPN between machines you control. What it does and does not protect:

- **Admin key and cluster token.** The admin key (`GPUPOOL_ADMIN_KEY`) protects the web UI and the admin API; the
  cluster token (`GPUPOOL_CLUSTER_TOKEN`) authenticates GPU servers to the coordinator and is part of the join
  command. Both are generated on first start and stored in `secrets.json` next to the database (the data volume in
  Docker); set your own values to override. Protect that file and the data volume.
- **API keys for `/v1`.** Clients of the OpenAI-compatible API send keys from `GPUPOOL_API_KEYS`. Without that
  variable `/v1` is open to anyone who can reach port 8080: set it.
- **The RPC port is unauthenticated.** llama.cpp's `ggml-rpc-server` (ports 9000 to 9999 on each GPU server) accepts
  any connection: whoever can reach it can allocate GPU memory and read or write tensors, and the traffic is not
  encrypted. Either set `GPUPOOL_RPC_FIREWALL=1` (the agent then adds iptables/ip6tables rules per RPC engine that
  allow only loopback, the head node and the agent's own address; it needs root and `NET_ADMIN`, for example
  `--cap-add NET_ADMIN`), or restrict 9000 to 9999 to cluster hosts with your own firewall, or keep the servers on a
  private network. The agent warns at startup when the firewall is off.
- **No built-in TLS.** The cluster token, admin key and API keys travel over plain HTTP. Put a TLS-terminating
  reverse proxy in front of the coordinator if anyone connects from outside a trusted network, and keep the servers
  on a private network or VPN.
- **Conversion does not run repository code.** Converting a Hugging Face model never downloads or executes the
  repository's Python files unless `allow_remote_code` is set explicitly for that job. The converter runs offline on
  a staging folder of whitelisted files, and gpupool deletes files only under its own conversion and cache folders.
  Treat `allow_remote_code` as running untrusted code on the coordinator.
- **The agent is privileged by design.** The agent container runs with `--pid host` and `--network host` (and
  `NET_ADMIN` when the RPC firewall is on) so it can see GPUs and processes and bind engine ports. Run it only on
  hosts you control, and do not expose its port (7070) beyond the coordinator.
- **Models are third-party data.** Model files are parsed by llama.cpp; only load models from sources you trust, and
  keep llama.cpp up to date.

See the Security section of [docs/QUICKSTART.en.md](docs/QUICKSTART.en.md) for the operational details.

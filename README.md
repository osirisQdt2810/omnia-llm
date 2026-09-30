# omnia-llm — a local LLM for Omnia that borrows one GPU only while it is working

This box has eight A5000s and other people working on them. A model server left running holds
a card between study sessions for nothing, so this one does not stay running: it starts on the
first request, and hands the card back after a configurable silence.

```
Anki / Omnia ──SSH tunnel──▶ gateway :8721   (always up, no GPU)
                                  │
                   /v1/chat/…     │     /v1/images/generations
                        ▼         │              ▼
                   vLLM :8722 ────┴──── imaged :8723
                        └──────── ONE card ──────┘
                                  │
                                  ▼  30 min with no request (each, separately)
                             stopped, card released
```

Two engines, one card, and each has its own idle timer — so the one you are not using gives
its memory back while the other keeps working.

## What is on the card

| | model | resident |
|---|---|---|
| text | `Qwen2.5-14B-Instruct-AWQ` (AWQ 4-bit, 4k context) | ~12.3 GiB |
| image | `stabilityai/sdxl-turbo` (4 steps, CPU offload) | ~1.5 GiB |
| | **both loaded** | **13.8 / 24.5 GiB** |

Measured, not estimated. Two earlier settings did not survive a real run and are worth knowing
about before you change them back:

* `gpu_memory_utilization = 0.48` left 0.49 GiB for a KV cache that needed 1.5 and vLLM refused
  to start at all;
* `0.55` with the image model resident sat at **23.9 of 24.5 GiB** — 97% of a card seven other
  people use, where one stray allocation of theirs fails both of us.

CPU offload on the image pipeline is what buys most of the headroom: ~1.5 GiB resident instead
of ~9.5, for about three seconds an image. A warm image is ~6.5s at 512×512.

## Why it is shaped like this

**The gateway and the engine are separate processes.** The gateway costs nothing to keep up
and can answer `/status` and `/v1/models` while no GPU is held — so checking on it, or a client
probing the connection, never wakes a card.

**`CUDA_VISIBLE_DEVICES`, not a device flag.** It makes the whole engine process unable to see
the other seven cards, including anything a library allocates at import time. A flag would
leave the door open for one stray allocation to land on somebody's job.

**The idle clock resets on the way in AND out of every request.** Only on the way in would let
the timer fire during a long generation and kill the engine answering it; only on the way out
would let a stalled request look idle.

**The two engines share ONE card, whichever is asked for first.** Each prefers the card the
other is on. That was one-directional at first — only the image engine shared — and the result
was that whichever engine started first claimed a card and the second took another: asking for
a picture before any text quietly cost two cards. Order of use is not something a user should
have to think about to keep a promise the service made.

**"Free" needs three signals to agree** — no compute process, under 512 MiB held, under 10%
utilization. Each catches a case the others miss: a job that has just started holds a context
before it allocates, a job between batches reports 0% while holding gigabytes, and a leaked
context holds memory with no live process. A wrong answer here costs a colleague their run.

**It picks the HIGHEST free index.** GPU 0 is where a display server and any unpinned script
land by default.

## Operating it

Everything this service owns lives under `~/workspaces/omnia-llm` — the venv, the logs, and
the ~10 GB weights cache (`hf-cache/`). The systemd unit is the one exception systemd forces:
the real file is `omnia-llm.service` here, and `~/.config/systemd/user/` holds a symlink to it.

```bash
systemctl --user start  omnia-llm      # the gateway (no GPU until a request arrives)
systemctl --user status omnia-llm
KEY=$(cat ~/workspaces/omnia-llm/api-key.txt)          # every route but /health needs it
curl -s -H "Authorization: Bearer $KEY" localhost:8721/status | python3 -m json.tool
curl -s -XPOST -H "Authorization: Bearer $KEY" localhost:8721/stop   # hand the card back NOW
journalctl --user -u omnia-llm -f                      # gateway log
tail -f ~/workspaces/omnia-llm/logs/vllm.log           # engine log
```

Everything you normally change is in `config.toml` — the model and
`idle_timeout_minutes` above all. Restart the gateway after editing it.

## Connecting Omnia

The gateway binds to `127.0.0.1` only; reach it over SSH rather than opening a port:

```bash
ssh -N -L 8721:127.0.0.1:8721 <your-host>
```

Then in Omnia: Smart Notes -> Usage & Keys -> Keys -> Add endpoint, with the Base URL, key and
models below (since Omnia #111 an endpoint is a named card; any number of them). The equivalent
single-slot `config/providers.toml` form still works:

```toml
[llm]
provider = "openai_compatible"

[llm.openai_compatible]
base_url    = "http://127.0.0.1:8721/v1"
api_key     = "<the contents of api-key.txt>"
text_model  = "omnia-local"
image_model = "sdxl"          # any name; the gateway has one image model
```

No new provider code, for either kind: Omnia's `openai_compatible` provider already posts to
`/chat/completions` and `/images/generations`, and the gateway answers both. It never learns
that one is vLLM and the other a diffusion pipeline.

The key is required. `/health` is the only endpoint that answers without it.

## Tests

```bash
source .venv/bin/activate && pip install pytest pytest-asyncio tomli && pytest
```

The tests cover the two things that hurt when wrong: which card counts as free, and whether
the engine is started once and released on time. They stub `nvidia-smi` and the subprocess, so
they run anywhere and need no GPU.

## Public access (ngrok) and tokens

The gateway is published over HTTPS by the `omnia-llm-ngrok` user service (outbound only, on the
account's fixed dev domain — the default random URL embeds this machine's IP, so never run ngrok
without `--url`). Every route but `/health` needs a token:

```bash
.venv/bin/python generate_auth_token.py <name>           # printed ONCE
.venv/bin/python generate_auth_token.py --list
.venv/bin/python generate_auth_token.py --revoke <name>  # immediate
systemctl --user status omnia-llm-ngrok
```

Tokens are stored hashed in `tokens.json` (0600); the old `api-key.txt` works as token `legacy`.
Ten wrong tokens in ten minutes lock that client out for fifteen (keyed on the X-Forwarded-For hop
ngrok adds; the client's own header is ignored). `POST /warm` starts the engines and returns at
once. Engines run under neutral process titles (`llm-engine`, `image-engine`).

# omnia-llm — a local LLM for Omnia that borrows one GPU only while it is working

This box has eight A5000s and other people working on them. A model server left running holds
a card between study sessions for nothing, so this one does not stay running: it starts on the
first request, and hands the card back after a configurable silence.

```
Anki / Omnia ──SSH tunnel──▶ gateway :8721  (always up, no GPU)
                                  │
                                  │ first request
                                  ▼
                             vLLM :8722  ── pinned to ONE free card
                                  │
                                  ▼  30 min with no request
                             stopped, card released
```

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

**"Free" needs three signals to agree** — no compute process, under 512 MiB held, under 10%
utilization. Each catches a case the others miss: a job that has just started holds a context
before it allocates, a job between batches reports 0% while holding gigabytes, and a leaked
context holds memory with no live process. A wrong answer here costs a colleague their run.

**It picks the HIGHEST free index.** GPU 0 is where a display server and any unpinned script
land by default.

## Operating it

```bash
systemctl --user start  omnia-llm      # the gateway (no GPU until a request arrives)
systemctl --user status omnia-llm
curl -s localhost:8721/status | python3 -m json.tool   # what is running, and every card
curl -s -XPOST localhost:8721/stop                     # hand the card back NOW
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

Then in Omnia's `config/providers.toml`:

```toml
[llm]
provider = "openai_compatible"

[llm.openai_compatible]
base_url = "http://127.0.0.1:8721/v1"
api_key  = "not-needed"      # the tunnel is the authentication
text_model = "omnia-local"
```

No new provider code: vLLM speaks the OpenAI API, which Omnia already has.

## Tests

```bash
source .venv/bin/activate && pip install pytest pytest-asyncio tomli && pytest
```

The tests cover the two things that hurt when wrong: which card counts as free, and whether
the engine is started once and released on time. They stub `nvidia-smi` and the subprocess, so
they run anywhere and need no GPU.

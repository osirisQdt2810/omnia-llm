# Any computer, no GPU

Runs a small text model on the processor with llama.cpp. It is slower than a GPU, but works on any
Mac or Linux computer.

**You need:** macOS or Linux, Python 3.10 or newer, about 2 GB of free disk, and llama.cpp's
`llama-server` program:

- **Mac:** `brew install llama.cpp`
- **Linux:** download a release from
  [llama.cpp's releases](https://github.com/ggml-org/llama.cpp/releases) and put `llama-server`
  on your PATH, or write its full path as `binary = "…"` under `[models.options]` in
  `configs/cpu-llamacpp.toml`.

## Set up

1. Install: `scripts/install.sh cpu`
2. Create a token for Omnia, and copy it now, because it is shown only once:
   `.venv/bin/omnia-llm token issue my-computer`

## Run

`scripts/run.sh configs/cpu-llamacpp.toml` (it serves on `http://127.0.0.1:8741`).

The first answer downloads the model (about 1 GB, kept in `.model-cache/`). On an M4 Mac that took
2.5 minutes, and short answers after it came back in about a third of a second. A slower
processor takes longer.

## Use it in Omnia

Add an endpoint in Omnia
([where](https://github.com/osirisQdt2810/omnia/blob/main/docs/guidance/local-server/README.md))
with:

| Field | Value |
|---|---|
| Base URL | `http://127.0.0.1:8741/v1` |
| API key | your token |
| Text model | `qwen2.5-1.5b` |

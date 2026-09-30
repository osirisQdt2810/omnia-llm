# Mac with Apple Silicon

Runs a small text model on the Mac's own GPU, for definitions, examples and translations. There
is no image model on the Mac yet.

**You need:** a Mac with an M-series chip, Python 3.10 or newer, and about 2 GB of free disk.

## Set up

1. Install: `scripts/install.sh apple`.
   If your `python3` is older than 3.10, name a newer one: `PYTHON=python3.12 scripts/install.sh apple`.
2. Create a token for Omnia, and copy it now, because it is shown only once:
   `.venv/bin/omnia-llm token issue my-mac`

## Run

`scripts/run.sh configs/mac-mlx-small.toml`

It serves on `http://127.0.0.1:8731` for as long as the terminal stays open, and Ctrl+C stops it.

The very first answer takes a minute or two, because the model is downloaded (about 0.9 GB, kept
in `.model-cache/`). After that the model loads in a few seconds whenever it has been idle, and
answers take under a second.

## Use it in Omnia

Add an endpoint in Omnia
([where](https://github.com/osirisQdt2810/omnia/blob/main/docs/guidance/local-server/README.md))
with:

| Field | Value |
|---|---|
| Base URL | `http://127.0.0.1:8731/v1` |
| API key | your token |
| Text model | `qwen2.5-1.5b` (press ↻ Load models and pick it) |

For better answers from a larger model, see [change or add models](models.md).

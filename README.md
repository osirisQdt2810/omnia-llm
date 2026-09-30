# omnia-llm

Your own AI models for [Omnia](https://github.com/osirisQdt2810/omnia), the Anki add-on. Run them
on a GPU server, a Mac or an ordinary computer, and Omnia writes and illustrates your cards with
them: no paid API, and your notes only go to machines you chose.

## What you get

- **One address and one token.** Omnia connects to omnia-llm the way it connects to any other
  provider, and lists the models it offers for you to pick.
- **Text and images.** Definitions, examples and translations from a text model, and pictures for
  your cards from an image model.
- **Only running when used.** A model starts with the first request and stops after 30 minutes
  without one, so nothing holds a GPU or memory between study sessions.
- **Access you control.** Every request needs a token that you issued, and a revoked token stops
  working immediately.

## Get started

| Your machine | Guide | What it runs |
|---|---|---|
| Mac with Apple Silicon | [Mac](docs/guidance/apple.md) | A small text model on the Mac's GPU |
| Linux with an NVIDIA GPU | [NVIDIA](docs/guidance/nvidia.md) | A 14B text model and an image model on one 24 GB GPU |
| Linux with an AMD GPU | [AMD](docs/guidance/rocm.md) | Preview: not yet tried on AMD hardware |
| Any Mac or Linux computer | [CPU](docs/guidance/cpu.md) | A small text model, no GPU needed |

Each guide ends with the three values to enter in Omnia. After that, you can
[use it from other computers](docs/guidance/remote-access.md) or
[change or add models](docs/guidance/models.md). To build a server of your own instead, see
[what Omnia expects from a server](https://github.com/osirisQdt2810/omnia/blob/main/docs/guidance/local-server/server-contract.md).

## In this repository

| Folder | What it holds |
|---|---|
| `configs/` | Ready-made setups, one per kind of machine |
| `scripts/` | Install and run |
| `deploy/` | Run it as a background service, and publish it over HTTPS |
| `docs/guidance/` | The guides above |
| `src/`, `tests/` | The server and its tests |

What it downloads and keeps stays inside this folder: models in `.model-cache/`, tokens and logs
in `state/`. Neither is ever committed.

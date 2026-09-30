# Linux with an NVIDIA GPU

Runs a 14B text model and an image model together on one 24 GB GPU. Each starts when it is first
asked for and stops after 30 minutes without requests, handing the GPU back, so it suits a machine
that other people use too.

**You need:** Linux, an NVIDIA GPU with at least 24 GB, a driver for CUDA 12.6 or newer, Python
3.10 to 3.12, and about 25 GB of free disk for the models.

## Set up

1. Install (this takes a while): `scripts/install.sh nvidia`
2. See the GPUs it can use: `.venv/bin/omnia-llm devices`.
   A GPU counts as free when nothing runs on it. omnia-llm takes one free GPU only, the
   highest-numbered one, and both models share it.
3. Create a token for each person or device, and copy it now, because it is shown only once:
   `.venv/bin/omnia-llm token issue <name>`

## Run

To try it: `scripts/run.sh configs/nvidia-24gb.toml` (it serves on `http://127.0.0.1:8721`).

To keep it running in the background and start it with the machine:

1. `cp deploy/omnia-llm.env.example deploy/omnia-llm.env`. This file names the config to run.
2. `ln -s "$PWD/deploy/systemd/omnia-llm.service" ~/.config/systemd/user/`
3. `systemctl --user daemon-reload`, then `systemctl --user enable --now omnia-llm`

The service expects this folder at `~/workspaces/omnia-llm`. If it is somewhere else, change the
paths in `deploy/systemd/omnia-llm.service`. To keep it running after you log out, your user
needs lingering turned on (`loginctl enable-linger`, which may need an administrator).

## Use it in Omnia

The server listens on this machine only. To reach it from the computer that runs Anki, publish
it first ([use it from other computers](remote-access.md)). Then add an endpoint in Omnia
([where](https://github.com/osirisQdt2810/omnia/blob/main/docs/guidance/local-server/README.md))
with:

| Field | Value |
|---|---|
| Base URL | `https://<your-address>/v1` |
| API key | your token |
| Text model | `omnia-local` |
| Image model | `sdxl-turbo` |

These are measured timings. After a pause, the first text answer takes about 1.5 minutes while the
model loads, and the answers after it about half a second. The first image takes about
20 seconds, and the next ones about 7.

In `nvidia-smi`, the running models show as `llm-engine` and `image-engine`.

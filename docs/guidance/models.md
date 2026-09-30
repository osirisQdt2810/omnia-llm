# Change or add models

What omnia-llm serves is set by a config file in `configs/`. Each `[[models]]` block in it is one
model, and Omnia lists it under its `id`. The ready-made configs name each model after itself,
such as `qwen2.5-14b-instruct-awq`.

| Setting | Meaning |
|---|---|
| `id` | The name Omnia shows, and sends back when it asks for this model. Letters, digits, `.`, `_` and `-` only |
| `kind` | `text` or `image` |
| `engine` | What runs the model (see below) |
| `port` | A free local port, used by this model only |
| `required_mib` | GPU memory the model needs, so it only starts where there is room (GPUs only) |
| `[models.options]` | The engine's own settings. `model` names the model to download |

| Engine | Runs on | What `model` is | Other options |
|---|---|---|---|
| `mlx` | Mac | An MLX model from Hugging Face, such as those under `mlx-community` | `max_tokens` |
| `vllm` | NVIDIA, AMD | A Hugging Face model | `max_model_len`, `gpu_memory_utilization` |
| `llamacpp` | Every platform | A local `.gguf` file, or `hf_repo = "<repo>:<quantization>"` instead. With both, `model` names a file in that repo | `ctx_size`, `gpu_layers`, `binary` |
| `diffusers` | NVIDIA, AMD (Mac and CPU untried) | A Hugging Face image model | `steps`, `cpu_offload` |

## Examples

- **A larger model on a Mac with 16 GB.** In `configs/mac-mlx-small.toml`, set
  `model = "mlx-community/Qwen2.5-7B-Instruct-4bit"` (about 4.3 GB), and `id` to its name,
  `qwen2.5-7b-instruct-4bit`. Then pick the new name in Omnia. Until you do, Omnia's requests
  under the old name still reach the text model.
- **A second text model.** Add another `[[models]]` block with its own `id` and `port`. Omnia then
  lists both.

## After a change

Check the file with `.venv/bin/omnia-llm models -c <config>`, which names any mistake, then
restart omnia-llm. A newly named model is downloaded on its first request, into `.model-cache/`.

Two other settings are worth knowing:

- `[server] idle_timeout_minutes`: how long a model may sit unused before it stops (default 30).
- `[devices] probe`: which hardware to use: `auto`, `nvidia`, `rocm`, `apple` or `cpu`.

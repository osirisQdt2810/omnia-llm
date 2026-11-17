# Change or add models

What omnia-llm serves is set by a config file in `configs/`. Each `[[models]]` block in it is one
model, and Omnia lists it under its `id`. The ready-made configs name each model after itself,
such as `qwen2.5-14b-instruct-awq`.

| Setting | Meaning |
|---|---|
| `id` | The name Omnia shows, and sends back when it asks for this model. Up to 64 letters, digits, `.`, `_` and `-`, starting with a letter or digit |
| `kind` | `text` or `image` |
| `engine` | What runs the model (see below) |
| `port` | A free local port, used by this model only |
| `required_mib` | GPU memory the model needs, so it only starts where there is room (GPUs only) |
| `[models.options]` | The engine's own settings. `model` names the model to download |
| `[models.when_cpu_busy]` | Optional. A different `required_mib`, and different options, for when the machine's CPUs are busy as the model starts (see below) |

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
- **Gemma 4 instead of Qwen, on the same 24 GB NVIDIA card.** `configs/nvidia-24gb-gemma4.toml`
  serves Gemma 4 26B-A4B (text) next to the image model. It gives better Vietnamese and fewer
  stray characters than the 14B Qwen, and starts in about 12 seconds. It runs on llama.cpp, so
  first build `llama-server` with CUDA into `bin/` (this takes a few minutes). The build needs the
  CUDA toolkit (`nvcc`, not only the driver) and the OpenSSL development headers: with libcurl
  switched off, llama.cpp downloads models through its own built-in HTTPS client.

  ```
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git bin/llama.cpp
  cmake -S bin/llama.cpp -B bin/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF
  cmake --build bin/llama.cpp/build --target llama-server -j
  ln -s llama.cpp/build/bin/llama-server bin/llama-server
  ```

  Then name the config in `deploy/omnia-llm.env`, restart omnia-llm, and pick
  `gemma-4-26b-a4b-it` as the text model in Omnia. The model (about 17 GB) is downloaded on its
  first request. Both models always stay on one card. When the machine's CPUs are idle as Gemma
  starts, part of it is kept in the machine's memory rather than on the card, which leaves room
  for the image model beside it; that needs about 10 GB of free system memory. When other people
  keep the CPUs busy, that part would slow Gemma to a crawl, so it takes the whole card instead,
  and the two models take turns on it: switching between text and pictures then costs a model
  start (about 12 seconds for Gemma, about 25 for the image model).

## After a change

Check the file with `.venv/bin/omnia-llm models -c <config>`, which names any mistake, then
restart omnia-llm. A newly named model is downloaded on its first request, into `.model-cache/`.

Other settings worth knowing:

- `[server] idle_timeout_minutes`: how long a model may sit unused before it stops (default 30).
  This file setting is only the default: the idle time can also be changed, between 1 minute and
  one day, from the endpoint card in Omnia, without a restart, and that choice is kept until it
  is changed again.
- `[devices] probe`: which hardware to use: `auto`, `nvidia`, `rocm`, `apple` or `cpu`.
- `[devices] single_device`: keep every model on one GPU (default `false`). Off, a model that
  does not fit beside the one already running takes a second free GPU. On, it never does: it
  starts beside the others when there is room, and otherwise waits until they have answered
  the requests they are working on, stops them, and starts in their place. The models then
  take turns, and the next request for a stopped one does the same in reverse. Use it on a
  machine you share, where taking a second GPU would take one from someone else. When no
  model is running, the first one to start picks a free GPU, as without this setting.
- `[server] cpu_busy_load` and `[models.when_cpu_busy]`: a model that keeps part of itself in
  the machine's memory computes that part on the CPU, which is fast while the CPUs are idle
  and very slow when other people keep them busy. A `when_cpu_busy` table under a model gives
  the settings to use instead when the CPUs are busy at the moment the model starts: its own
  `required_mib`, and an `options` table whose entries replace the model's options of the same
  name (a list is replaced whole). The CPUs count as busy when the machine's load over the
  last minute, per CPU, is at least `cpu_busy_load` (default `0.7`). A model already running
  keeps the settings it started with until it stops. The log names the settings each start
  used and the load that decided it, and `/status` shows them as each model's `profile`.

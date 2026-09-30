# Linux with an AMD GPU (preview)

omnia-llm finds AMD GPUs through ROCm and can run the ROCm builds of vLLM (text) and PyTorch
(images). This path has not been tried on AMD hardware yet, so expect to adjust versions.

**You need:** Linux with ROCm installed (`rocm-smi` works in a terminal), an AMD GPU with enough
memory for your model, and Python 3.10 to 3.12.

## Set up

1. Create the environment with `python3 -m venv .venv`. Then install the ROCm builds of PyTorch
   and vLLM into it, following AMD's instructions for your ROCm version.
2. Add omnia-llm to it: `.venv/bin/pip install -e .`
3. Check that it sees your GPUs: `.venv/bin/omnia-llm devices --probe rocm`
4. Make a config. Copy `configs/nvidia-24gb.toml` to `configs/rocm.toml`, and under `[devices]` set
   `probe = "rocm"`. Check it with `.venv/bin/omnia-llm models -c configs/rocm.toml`.
5. Create a token: `.venv/bin/omnia-llm token issue <name>`

## Run and use it

Run it as on NVIDIA, with `scripts/run.sh configs/rocm.toml`, then
[connect Omnia](nvidia.md#use-it-in-omnia) the same way.

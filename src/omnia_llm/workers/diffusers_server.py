"""A text-to-image server speaking the OpenAI ``/v1/images/generations`` shape.

Run as its OWN process, like vLLM and for the same reason: killing a process is the only
reliable way to give a GPU's memory back. Unloading a diffusion pipeline inside a long-lived
server leaves the allocator fragmented and the card still spoken for, which would defeat the
idle timer this whole service exists for.

Diffusion is not an LLM and vLLM cannot serve it — but Omnia does not need to know that. Its
``openai_compatible`` provider already posts to ``/images/generations`` and reads
``data[0].b64_json``, so speaking that shape here means the add-on needs no new provider, no
new code, and no idea which kind of model is behind the URL.

Started by the gateway already pinned to one device (``CUDA_VISIBLE_DEVICES`` /
``HIP_VISIBLE_DEVICES``), so this module never chooses a GPU and cannot take one it was not given.
``--device`` only says which KIND it is: ``nvidia`` / ``rocm`` (torch's "cuda"), ``apple`` (MPS)
or ``cpu``.
"""

from __future__ import annotations

import argparse
import base64
import io
import threading
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="omnia image engine")

#: Loaded once on first use and kept for the process's life. The gateway's idle timer is what
#: ends that life, so there is nothing to unload here — the whole process goes.
_pipeline: Optional[Any] = None
_lock = threading.Lock()
_model_id = ""
_steps = 4
_cpu_offload = True
_backend = "nvidia"


class _Request(BaseModel):
    """The subset of the OpenAI images payload that means anything for a local pipeline."""

    prompt: str = ""
    model: str = ""
    size: str = "1024x1024"
    n: int = 1
    response_format: str = "b64_json"


def _load() -> Any:
    """Build the pipeline, once, under a lock.

    Under a lock because two concurrent first requests would otherwise each load several
    gigabytes onto the same card — which on a shared box is how you take somebody else's
    memory by accident.
    """
    global _pipeline
    with _lock:
        if _pipeline is not None:
            return _pipeline
        import torch
        from diffusers import AutoPipelineForText2Image

        gpu = _backend in ("nvidia", "rocm")  # ROCm builds of torch answer as "cuda"
        _pipeline = AutoPipelineForText2Image.from_pretrained(
            _model_id,
            # fp16 where it runs well (a GPU or Apple's MPS); a CPU wants fp32.
            torch_dtype=torch.float32 if _backend == "cpu" else torch.float16,
            # fp16 weights where the repo has them: half the download and half the VRAM, which
            # is the difference between sharing this card and needing a second one.
            variant=None if _backend == "cpu" else "fp16",
            use_safetensors=True,
        )
        if not gpu:
            _pipeline.to("mps" if _backend == "apple" else "cpu")
        elif _cpu_offload:
            # Components live in system RAM and move to the card only for the step that needs
            # them. ~4 GiB resident instead of ~9.5, at a few seconds per image.
            #
            # On by default because of what this machine is: the card is shared with seven
            # other people and the text model is on it too. Holding 9.5 GiB continuously to
            # save three seconds on an occasional picture is the wrong side of that trade. An
            # operator whose card is their own can turn it off.
            _pipeline.enable_model_cpu_offload()
        else:
            _pipeline.to("cuda")
        # No NSFW filter pass: it is a second model on a card with a budget, and the prompts
        # here are vocabulary words from the user's own deck.
        if hasattr(_pipeline, "safety_checker"):
            _pipeline.safety_checker = None
        return _pipeline


def _size(value: str) -> tuple[int, int]:
    """Parse ``"1024x1024"``, falling back rather than failing on something unparseable.

    A bad size is not worth refusing a generation over: the caller gets a picture at the
    default size, which is what it would have asked for if it had not been mistaken.
    """
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except Exception:  # noqa: BLE001 - any malformed value lands on the default
        return 1024, 1024
    # Rounded to the multiple of 8 the UNet requires; a stray 1023 would raise deep inside it.
    return max(256, width - width % 8), max(256, height - height % 8)


@app.get("/health")
def health() -> JSONResponse:
    """Up, but not necessarily loaded — the gateway polls this to know the process is alive."""
    return JSONResponse({"ok": True, "loaded": _pipeline is not None})


@app.post("/v1/images/generations")
def generate(request: _Request) -> JSONResponse:
    """One image, returned as base64 in the OpenAI envelope."""
    if not request.prompt.strip():
        return JSONResponse(
            {"error": {"message": "prompt is required", "type": "invalid_request"}},
            status_code=400,
        )
    width, height = _size(request.size)
    try:
        pipeline = _load()
        image = pipeline(
            prompt=request.prompt,
            width=width,
            height=height,
            num_inference_steps=_steps,
            guidance_scale=0.0 if _steps <= 4 else 5.0,
        ).images[0]
    except Exception as exc:  # noqa: BLE001 - boundary: one clear message, never a traceback
        return JSONResponse(
            {"error": {"message": f"image generation failed: {exc}", "type": "engine_error"}},
            status_code=500,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return JSONResponse(
        {
            "created": 0,
            "data": [{"b64_json": base64.b64encode(buffer.getvalue()).decode("ascii")}],
        }
    )


def main() -> None:
    global _model_id, _steps, _cpu_offload, _backend

    parser = argparse.ArgumentParser(description="omnia image engine")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    # SDXL-Turbo and FLUX-schnell are distilled to a handful of steps; a full SDXL wants ~30.
    # Exposed because it is the one knob that trades seconds for quality, and the right answer
    # depends on which weights the operator pointed this at.
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--no-cpu-offload",
        action="store_true",
        help="keep the whole pipeline resident on the GPU (faster, ~9.5 GiB instead of ~4)",
    )
    parser.add_argument("--device", default="nvidia",
                        choices=("nvidia", "rocm", "apple", "cpu"))
    args = parser.parse_args()
    _model_id, _steps, _backend = args.model, args.steps, args.device
    _cpu_offload = not args.no_cpu_offload
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

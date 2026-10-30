"""Remote text encoder: run the prompt encoder in a different process, on a different GPU.

A torch build drives one GPU vendor (ROCm *or* CUDA), so a Radeon and an NVIDIA card
cannot share a process. This splits the pipeline at its natural seam instead: a
``thenoise text-encoder-serve`` worker holds the encoder on one backend, and the main
process (built for the other backend) asks it for embeddings over HTTP.

Wire format (kept pickle-free, since the worker may listen beyond localhost):

    POST /encode   JSON {"prompt": str, "images": [base64 PNG, ...]}
      -> safetensors bytes holding the ``hidden`` tensor, slots in the ``X-Slots`` header
    GET  /health   {"status": "ok"}

Only Qwen-Image 2.1's encoder (``(prompt, images) -> (embeddings, slots)``) is served.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import threading
import urllib.error
import urllib.request
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
from pydantic import BaseModel
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

logger = logging.getLogger(__name__)

SLOTS_HEADER = "X-Slots"
TENSOR_NAME = "hidden"
_DEFAULT_TIMEOUT_S = 600.0  # the first request pays for lazy kernels/compilation


class RemoteEncoderError(RuntimeError):
    """The remote encoder could not be reached or refused the request."""


class EncodeBody(BaseModel):
    prompt: str
    images: List[str] = []


# ------------------------------------------------------------------ wire helpers
def encode_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_image(data: str) -> Image.Image:
    image = Image.open(io.BytesIO(base64.b64decode(data)))
    image.load()
    return image


def pack_embeddings(hidden: torch.Tensor, slots: Sequence[int]) -> Tuple[bytes, dict]:
    """``(body, headers)`` carrying ``hidden`` and its image slots."""
    body = save_tensors({TENSOR_NAME: hidden.detach().contiguous().cpu()})
    return body, {SLOTS_HEADER: json.dumps([int(s) for s in slots])}


def unpack_embeddings(body: bytes, slots_header: str) -> Tuple[torch.Tensor, List[int]]:
    return load_tensors(body)[TENSOR_NAME], [int(s) for s in json.loads(slots_header)]


# ---------------------------------------------------------------------- client
class RemoteTextEncoder:
    """Stands in for ``QwenImage21TextEncoder``: ``(prompt, images) -> (embeddings, slots)``.

    The embeddings come back on ``device`` in ``dtype``, ready for the DiT.
    """

    def __init__(
        self,
        url: str,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        timeout: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self.url = url.rstrip("/")
        self.device = torch.device(device)
        self.dtype = dtype
        self.timeout = timeout

    def check(self) -> None:
        """Fail early, at model load, if the worker is not up."""
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=10) as resp:
                resp.read()
        except (urllib.error.URLError, OSError) as exc:
            raise RemoteEncoderError(
                f"text encoder worker not reachable at {self.url}: {exc}. Start it with "
                "'thenoise text-encoder-serve' first."
            ) from exc

    def __call__(
        self, prompt: str, images: Optional[Sequence[Image.Image]] = None
    ) -> Tuple[torch.Tensor, List[int]]:
        payload = json.dumps({
            "prompt": prompt,
            "images": [encode_image(img) for img in images or []],
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}/encode", data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                slots_header = resp.headers.get(SLOTS_HEADER)
                body = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RemoteEncoderError(
                f"text encoder worker returned {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RemoteEncoderError(
                f"text encoder worker not reachable at {self.url}: {exc}"
            ) from exc
        if slots_header is None:
            raise RemoteEncoderError(f"text encoder worker reply lacks {SLOTS_HEADER}")
        hidden, slots = unpack_embeddings(body, slots_header)
        return hidden.to(device=self.device, dtype=self.dtype), slots


# ---------------------------------------------------------------------- server
def create_encoder_app(encoder):
    """A FastAPI app serving ``encoder`` (anything callable as ``(prompt, images)``)."""
    from fastapi import FastAPI, HTTPException, Response

    from thenoise.inference import inference

    app = FastAPI(title="thenoise text encoder")
    lock = threading.Lock()  # one encode at a time on the device

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/encode")
    def encode(body: EncodeBody):
        try:
            images = [decode_image(data) for data in body.images]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"bad image: {exc}") from exc
        with lock, inference():
            hidden, slots = encoder(body.prompt, images)
        payload, headers = pack_embeddings(hidden, slots)
        return Response(
            content=payload, media_type="application/octet-stream", headers=headers
        )

    return app


def run_encoder_server(args) -> None:
    """Load the encoder on ``args.device`` and serve it until interrupted."""
    import uvicorn

    from thenoise.dit.qwen_image21.encoder import load_qwen_image21_text_encoder
    from thenoise.inference import inference
    from thenoise.utils.device import release_device_memory

    logging.basicConfig(level=logging.INFO)
    logger.info("Loading Qwen-Image 2.1 text encoder from %s on %s",
                args.text_encoder, args.device)
    with inference():
        encoder = load_qwen_image21_text_encoder(
            args.text_encoder, dtype=torch.bfloat16, device=args.device
        )
    release_device_memory(args.device)
    app = create_encoder_app(encoder)
    print(f"thenoise text encoder serving on http://{args.host}:{args.port} ({args.device})")
    uvicorn.run(app, host=args.host, port=args.port)


__all__ = [
    "RemoteTextEncoder",
    "RemoteEncoderError",
    "create_encoder_app",
    "run_encoder_server",
]

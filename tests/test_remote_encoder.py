"""Remote text encoder: wire format, server, client, and the adapter/pipeline wiring.

No weights and no network: the client's ``urlopen`` is routed to the app in-process.
"""
from __future__ import annotations

import io
import urllib.error

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from thenoise import remote_encoder
from thenoise.cli import build_parser, require_text_encoder
from thenoise.models import QwenImage21Model
from thenoise.models.base import DiffusionModel
from thenoise.models.config import EncodePromptArgs
from thenoise.remote_encoder import (
    RemoteEncoderError,
    RemoteTextEncoder,
    create_encoder_app,
    pack_embeddings,
    unpack_embeddings,
)
from thenoise.runtime import ModelPaths, Runtime, Settings


class FakeEncoder:
    """Deterministic stand-in: one row per prompt character, one extra per image."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt, images=None):
        images = list(images or [])
        self.calls.append((prompt, [img.size for img in images]))
        rows = len(prompt) + len(images)
        hidden = torch.arange(rows * 4, dtype=torch.float32).reshape(1, rows, 4)
        return hidden.to(torch.bfloat16), list(range(len(images)))


class _Response(io.BytesIO):
    def __init__(self, body, headers):
        super().__init__(body)
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture
def worker(monkeypatch):
    """The served FakeEncoder, with the client's HTTP calls routed into the app."""
    encoder = FakeEncoder()
    http = TestClient(create_encoder_app(encoder))

    def urlopen(request, timeout=None):
        if isinstance(request, str):
            url, data = request, None
        else:
            url, data = request.full_url, request.data
        path = url.split("//", 1)[1].split("/", 1)[1]
        if data is None:
            resp = http.get("/" + path)
        else:
            resp = http.post("/" + path, content=data,
                             headers={"Content-Type": "application/json"})
        if resp.status_code != 200:
            raise urllib.error.HTTPError(
                url, resp.status_code, "err", {}, io.BytesIO(resp.content))
        return _Response(resp.content, resp.headers)

    monkeypatch.setattr(remote_encoder.urllib.request, "urlopen", urlopen)
    return encoder


def _client(**kw):
    return RemoteTextEncoder("http://worker:8100/", device="cpu", dtype=torch.float32, **kw)


def test_embeddings_round_trip_through_the_wire_format():
    hidden = torch.randn(1, 5, 8).to(torch.bfloat16)

    body, headers = pack_embeddings(hidden, [3, 7])
    out, slots = unpack_embeddings(body, headers[remote_encoder.SLOTS_HEADER])

    assert torch.equal(out, hidden) and out.dtype == torch.bfloat16
    assert slots == [3, 7]


def test_client_returns_embeddings_on_the_target_device_and_dtype(worker):
    hidden, slots = _client()("abc")

    assert hidden.shape == (1, 3, 4) and hidden.dtype == torch.float32
    assert slots == []
    assert worker.calls == [("abc", [])]


def test_references_reach_the_worker_as_images_and_slots_come_back(worker):
    images = [Image.new("RGB", (32, 64), "red"), Image.new("RGB", (16, 16), "blue")]

    hidden, slots = _client()("hi", images)

    assert worker.calls == [("hi", [(32, 64), (16, 16)])]
    assert hidden.shape[1] == 2 + 2 and slots == [0, 1]


def test_check_passes_against_a_live_worker(worker):
    _client().check()


def test_a_bad_image_is_a_400_not_a_crash():
    http = TestClient(create_encoder_app(FakeEncoder()))

    resp = http.post("/encode", json={"prompt": "p", "images": ["not-base64-png"]})

    assert resp.status_code == 400


def test_a_worker_error_is_reported_with_its_detail(monkeypatch):
    def refuse(request, timeout=None):
        raise urllib.error.HTTPError("u", 500, "boom", {}, io.BytesIO(b"kaput"))

    monkeypatch.setattr(remote_encoder.urllib.request, "urlopen", refuse)

    with pytest.raises(RemoteEncoderError, match="500.*kaput"):
        _client()("p")


def test_an_unreachable_worker_says_how_to_start_it(monkeypatch):
    def down(request, timeout=None):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(remote_encoder.urllib.request, "urlopen", down)

    with pytest.raises(RemoteEncoderError, match="text-encoder-serve"):
        _client().check()


# ------------------------------------------------------------ adapter wiring
def test_the_adapter_encodes_through_the_remote_and_needs_no_local_room(worker):
    model = object.__new__(QwenImage21Model)
    model.device, model.dtype = "cpu", torch.float32
    model.text_encoder = _client()

    cond = model.encode_prompt(EncodePromptArgs(prompt="abcd", guidance_scale=1.0))

    assert model.remote_text_encoder
    assert cond.cond.shape == (1, 4, 4) and cond.null is None


def test_guidance_asks_the_worker_for_the_negative_prompt_too(worker):
    model = object.__new__(QwenImage21Model)
    model.device, model.dtype = "cpu", torch.float32
    model.text_encoder = _client()

    cond = model.encode_prompt(
        EncodePromptArgs(prompt="ab", negative_prompt="xyz", guidance_scale=4.0))

    assert [c[0] for c in worker.calls] == ["ab", "xyz"]
    assert cond.null.shape == (1, 3, 4)


def test_a_local_encoder_is_not_remote():
    assert DiffusionModel.remote_text_encoder.fget(object()) is False


def test_only_adapters_that_support_it_accept_a_remote_encoder(monkeypatch):
    class Local:
        name = "local"
        REMOTE_TEXT_ENCODER = False

    monkeypatch.setattr("thenoise.models.resolve", lambda path: Local)
    runtime = Runtime(Settings(device="cpu"))

    with pytest.raises(ValueError, match="remote text encoder"):
        runtime.load(ModelPaths("d", "v", "", text_encoder_url="http://w:1"))


# ------------------------------------------------------------------------ CLI
def test_cli_parses_the_encoder_worker_and_the_remote_flag():
    worker = build_parser().parse_args(["text-encoder-serve", "--text-encoder", "t.st"])
    assert (worker.host, worker.port, worker.device) == ("127.0.0.1", 8100, "cuda")

    gen = build_parser().parse_args(
        ["generate", "--dit", "d", "--vae", "v", "--text-encoder-url", "http://w:8100",
         "--prompt", "x"])
    assert gen.text_encoder_url == "http://w:8100" and gen.text_encoder is None
    require_text_encoder(gen)


def test_a_model_without_any_encoder_source_is_refused():
    args = build_parser().parse_args(
        ["generate", "--dit", "d", "--vae", "v", "--prompt", "x"])

    with pytest.raises(SystemExit, match="--text-encoder"):
        require_text_encoder(args)

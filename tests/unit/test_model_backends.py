"""Backend selection and the local MLX adapter's prompt routing."""

import asyncio

import numpy as np
import pytest

from oxr.config.settings import LayoutAnalyzeModelConfig, OXRModelConfig, Settings
from oxr.model import OXRModelBackend, build_model
from oxr.model.base import ModelOutput
from oxr.model.client import OXRModelClient
from oxr.model.local_mlx import LocalMLXModel
from oxr.model.mlx.runner import Generation
from oxr.model.prompts import (
    PROMPT_CODE_RECOGNITION,
    PROMPT_FORMULA_RECOGNITION,
    PROMPT_TABLE_RECOGNITION,
    PROMPT_TEXT_RECOGNITION,
)

IMAGE = np.zeros((8, 8, 3), dtype=np.uint8)


def test_default_backend_is_remote():
    backend = build_model(OXRModelConfig(url="http://127.0.0.1:8080/v1"))
    assert isinstance(backend, OXRModelClient)
    assert isinstance(backend, OXRModelBackend)


def test_mlx_backend_selected_by_config(tmp_path):
    backend = build_model(OXRModelConfig(backend="mlx", model_path=str(tmp_path)))
    assert isinstance(backend, LocalMLXModel)
    assert isinstance(backend, OXRModelBackend)


def test_server_validation_requires_url_for_remote():
    settings = Settings(
        layout_analyze_model=LayoutAnalyzeModelConfig(url="http://127.0.0.1:8081"),
        oxr_model=OXRModelConfig(backend="remote"),
    )
    with pytest.raises(ValueError, match="oxr_model.url"):
        settings.validate_for_server()


def test_server_validation_requires_path_for_mlx():
    settings = Settings(
        layout_analyze_model=LayoutAnalyzeModelConfig(url="http://127.0.0.1:8081"),
        oxr_model=OXRModelConfig(backend="mlx"),
    )
    with pytest.raises(ValueError, match="oxr_model.model_path"):
        settings.validate_for_server()


def test_server_validation_accepts_existing_mlx_checkpoint(tmp_path):
    settings = Settings(
        layout_analyze_model=LayoutAnalyzeModelConfig(url="http://127.0.0.1:8081"),
        oxr_model=OXRModelConfig(backend="mlx", model_path=str(tmp_path)),
    )
    settings.validate_for_server()


class FakeRunner:
    def __init__(self):
        self.calls = []

    def generate(self, image, prompt, max_tokens, repetition_penalty):
        self.calls.append((prompt, max_tokens, repetition_penalty))
        return Generation(text="ok", finish_reason="length")


def _local(tmp_path) -> tuple[LocalMLXModel, FakeRunner]:
    model = LocalMLXModel(OXRModelConfig(backend="mlx", model_path=str(tmp_path)))
    runner = FakeRunner()
    model._load = lambda: runner
    return model, runner


@pytest.mark.parametrize(
    ("method", "prompt"),
    [
        ("recognize_text", PROMPT_TEXT_RECOGNITION),
        ("recognize_formula", PROMPT_FORMULA_RECOGNITION),
        ("recognize_table", PROMPT_TABLE_RECOGNITION),
        ("recognize_code", PROMPT_CODE_RECOGNITION),
    ],
)
def test_local_backend_routes_prompts(tmp_path, method, prompt):
    model, runner = _local(tmp_path)
    result = asyncio.run(getattr(model, method)(IMAGE))
    assert isinstance(result, ModelOutput)
    assert result == "ok"
    assert result.finish_reason == "length"
    assert runner.calls[0][0] == prompt


def test_local_backend_honours_max_tokens(tmp_path):
    model, runner = _local(tmp_path)
    asyncio.run(model.request_reading_order("order the boxes", IMAGE, 64))
    assert runner.calls[0][1] == 64


def test_local_backend_serialises_requests(tmp_path):
    model, _ = _local(tmp_path)
    assert model.get_request_semaphore()._value == 1

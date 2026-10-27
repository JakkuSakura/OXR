# OXR local setup notes (this machine)

OXR = the 1.3B document-parsing model from SoMark described in the WeChat article
(https://mp.weixin.qq.com/s/BBNjdqN1YBJSLf7ApUHCHQ).
Upstream repo: https://github.com/SoMarkAI/OXR — weights: https://huggingface.co/SoMarkAI/OXR-1.0

## Environments

| Item | Location |
|-|-|
| Parsing API venv | `~/Dev/OXR/.venv` — `oxr` CLI + `somark` client |
| MLX venv | `~/Dev/OXR/.venv-mlx` — `oxr` (editable), mlx, torch |
| Weights (~3.1GB) | `~/Dev/OXR/models/OXR-1.0/` (gitignored) |
| Config | `~/Dev/OXR/config.yaml` (copy of `config.example.yaml`, gitignored) |

```bash
cd ~/Dev/OXR
uv venv .venv --python 3.12
uv pip install -p .venv/bin/python -c docker/requirements.allinone.txt . somark

uv venv .venv-mlx --python 3.12
uv pip install -p .venv-mlx/bin/python -c docker/requirements.allinone.txt -e ".[mlx]" torch torchvision pytest
```

`.venv-mlx` has the repo installed **editable** (`-e .`), so CLI changes are picked
up without reinstalling. If `oxr serve` reports "No such option", this is the cause.

## Parsing with the public free API (no GPU, no local model)

Use the `.cn` host: `free-oxr.somark.ai` currently answers
`404 … the free API is available at https://free-oxr.somark.cn/api/v1`.

```bash
cd ~/Dev/OXR
.venv/bin/somark --base-url https://free-oxr.somark.cn/api/v1 --api-key dummy \
  parse ~/Documents/some.pdf --formats md,json --out ./out/
```

`--out` must already exist. One task at a time per client; a second call while one
is running returns `429` / `code 1133`. Not for production use.

## Running locally on this Mac (MLX backend)

`oxr_model.backend: mlx` runs recognition *inside* `oxr serve` — no model service,
no shim, no extra port. Layout analysis is still upstream's separate service.

```bash
cd ~/Dev/OXR

# layout analysis (upstream service, unchanged), torch on CPU here
MODEL_PATH="$PWD/models/OXR-1.0/layout_analysis/last.pth" DFINE_DEVICE=cpu DFINE_CUDA_GRAPH=0 \
  .venv-mlx/bin/python -m uvicorn dfine_la.api:app --app-dir services/layout_analyze \
  --host 127.0.0.1 --port 8081 --workers 1

# parsing API + demo, recognition in-process on Metal
.venv-mlx/bin/oxr serve --config config.yaml --host 127.0.0.1 --port 8010 --workers 1 \
  --oxr-model-backend mlx --oxr-model-path "$PWD/models/OXR-1.0"
```

`:8010` instead of `:8000` because `omlx-server` already listens on 8000 here.
The equivalent config.yaml form:

```yaml
oxr_model:
  backend: mlx
  model_path: ./models/OXR-1.0
  dtype: bfloat16
```

```bash
curl -X POST http://127.0.0.1:8010/v1/parse/sync \
  -F file=@doc.pdf -F api_key=dummy -F output_formats=markdown
open http://127.0.0.1:8010/demo
```

The model loads during server startup (`await model_client.warmup()`), ~5 s.

## Where the code lives

| File | Role |
|-|-|
| `oxr/model/base.py` | `OXRModelBackend` protocol + `ModelOutput` (the abstraction) |
| `oxr/model/__init__.py` | `build_model()`: picks a backend from `oxr_model.backend` |
| `oxr/model/client.py` | `remote` backend — OpenAI-compatible HTTP client (upstream code) |
| `oxr/model/local_mlx.py` | `mlx` backend — in-process, single worker thread |
| `oxr/model/mlx/model.py` | SigLIP2-NaViT tower, 65532-way visual vocabulary head, Qwen3-0.6B |
| `oxr/model/mlx/preprocess.py` | smart_resize + bicubic + 2×2-block patchify |
| `oxr/model/mlx/runner.py` | prompt assembly, greedy decode with repetition penalty |
| `tests/mlx/` | manual parity/preprocessing checks (torch, weights required) |

## Checks

```bash
.venv-mlx/bin/python -m pytest tests/unit -q          # 628 tests, no weights
.venv-mlx/bin/python -m tests.mlx.parity <image.png>  # needs weights + torch
```

### Validation against the float32 PyTorch reference

- patch vectors: identical to transformers (max |Δ| = 0)
- positional embeddings: 1.4e-5
- visual soft tokens: 1.0e-5
- prompt logits (876 tokens): 5.9e-4, same argmax
- end-to-end markdown equals the free API's output for the same PDF

### Speed on this machine (M-series, 64GB)

| Workload | Time |
|-|-|
| server start incl. model load | ~25 s |
| one 1025×585 crop, ~60 output tokens | ~1.6 s |
| 1-page PDF, layout + 6 blocks | ~4–11 s |
| 3-page dense text PDF | ~52 s |

## Things that bit me (worth knowing if you touch the backend)

- `fullatt_block_indexes: null` means **full** attention in every vision layer; the
  window permutation is still applied but is a no-op under full attention.
- `mx.fast.rope` did not follow the `(batch, seq, heads, dim)` axis convention I
  expected, so RoPE is applied explicitly.
- Indicator tokens (-301..-304) map to the **tail** of the visual vocabulary
  (65532..65535); negative Python indices silently pick the wrong rows.
- The visual head's LayerNorm uses torch's default eps (1e-5), not the config's 1e-6.
- MLX arrays belong to the thread that created them (weights included), hence the
  dedicated worker thread that also loads the model.
- The pipeline renders PDFs at 150 DPI; small glyphs read poorly, mitigated by the
  model's own upscaling of crops to ≥448².

## Upstream path: Linux + NVIDIA

Unchanged — `docker/Dockerfile.allinone`, or `services/oxr/start_vllm.sh` with
`pip install vllm==0.25.1 transformers==5.13.1`, then `oxr serve` with
`oxr_model.url` pointing at it.

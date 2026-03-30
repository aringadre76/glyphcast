# glyphcast

GPU-accelerated pipeline that turns videos and GIFs into clean ASCII art using edge detection and per-cell character classification (with a CPU-friendly fallback path).

## Features

- **Edge backends**: PyTorch-ready `dexined`, `hed`, and `sobel`. Sobel is used when checkpoints are missing or the `[edge]` extra is not installed.
- **Glyphs**: Synthetic glyph dataset generation, a lightweight char CNN, and a Random Forest baseline for offline experiments. Runtime `glyph_mode` can use the CNN with template fallback when weights or CUDA are unavailable.
- **Ingestion**: OpenCV for video; dedicated GIF decoding.
- **Temporal smoothing**: Optional motion-aware smoothing to cut flicker between frames.
- **Background suppression**: Optional runtime tuning (`background_suppression` and related thresholds in YAML) to quiet low-confidence background cells.
- **Outputs**: Terminal playback, stacked text export, and MP4 (`render.overlay_mode`: `ascii_only`, `blended`, or `source_tinted`). For blends, the source frame is resized to the ASCII canvas when resolutions differ so compositing stays shape-safe.
- **CLI**: Typer with presets (`default`, `fast`, `high_quality`). `render` and `benchmark` print a short runtime summary (device, edge backend, glyph mode, checkpoint paths) so you can see what actually ran.

## Requirements

- Python **3.11+** (see `pyproject.toml`).

## Project layout

```text
glyphcast/                 # installable package
  commands/                # CLI: render, train-chars, benchmark, download-models
  io/                      # Video and GIF decoding
  models/                  # Edge backends and character models
  pipeline/                # Preprocess, tiling, mapping, temporal smoothing
  render/                  # Terminal, text export, video overlay, compositing
  training/                # Synthetic glyphs and training
  utils/                   # Profiling helpers
configs/                   # default.yaml, fast.yaml, high_quality.yaml
scripts/                   # Edge model helpers
tests/
```

## Quick start

1. Create a Python 3.11+ environment.

2. Install (pick extras as needed):

```bash
python3 -m pip install -e .
# Development, training, and neural edge backends:
python3 -m pip install -e ".[dev,train,edge]"
```

3. Render a GIF to text:

```bash
glyphcast render giphy.gif --mode text --output artifacts/renders/giphy.txt
# equivalent:
python3 -m glyphcast.cli render giphy.gif --mode text --output artifacts/renders/giphy.txt
```

4. Download edge checkpoints (optional; enables DexiNed/HED when `[edge]` and PyTorch are available):

```bash
glyphcast download-models --edge all
```

Character CNN weights live under `artifacts/models/chars/` by default. Train your own with `train-chars` or point `runtime.char_model_path` in YAML to your checkpoint.

## Configuration

- **Presets** (`--preset`): `default`, `fast`, `high_quality` — YAML files in `configs/`.
- **Charset** (`train-chars --charset` or `runtime.charset` in YAML): `minimal`, `balanced`, or `dense` (three distinct progressive charsets), or a custom string.
- **Runtime**: `device`, `edge_backend`, `glyph_mode`, smoothing, background suppression, batch size, and fallbacks are all driven from the preset YAML; override by editing a copy or adding a new preset file.

## Commands

| Command | Purpose |
|--------|---------|
| `glyphcast render <input> --mode {terminal,text,video} [--output PATH] [--preset NAME]` | Full pipeline on a video or GIF. |
| `glyphcast benchmark <input> [--preset NAME]` | One-frame timing and metadata; prints runtime summary. |
| `glyphcast train-chars [--charset minimal\|balanced\|dense] [--fonts PATH] [--preset NAME]` | Train the char CNN. |
| `glyphcast download-models [--edge dexined\|hed\|all] [--destination DIR]` | Fetch edge weights (default: DexiNed only; destination defaults to `artifacts/models/edge/`). |

Use `glyphcast <command> --help` for full options.

## Development

```bash
python3 -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```

## License

MIT — see `pyproject.toml`.

## Notes

- Without `[edge]` or without downloaded edge weights, the pipeline falls back to Sobel while keeping the same CLI and config fields.
- For neural edges, install `[edge]` and run `download-models`; char inference still expects checkpoints under `artifacts/models/chars/` when the config uses a CNN-backed `glyph_mode`.

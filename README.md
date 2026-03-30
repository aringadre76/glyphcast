# glyphcast

GPU-accelerated ML pipeline that converts videos and GIFs into clean, noise-free ASCII art using edge detection and character classification.

## Features

- PyTorch-ready edge detection with `dexined`, `hed`, and `sobel` backends (Sobel used when checkpoints are missing or `[edge]` is not installed).
- Synthetic glyph dataset generation and a lightweight CNN for ASCII character classification, plus a Random Forest baseline for offline experiments.
- OpenCV-based video ingestion and GIF decoding.
- Motion-aware temporal smoothing to reduce flicker between frames.
- Output modes: terminal playback, text export, and MP4 overlay (`ascii_only`, `blended`, or `source_tinted` in config).
- Typer CLI with presets (`default`, `fast`, `high_quality`) and runtime summaries (device, edge backend, glyph mode, checkpoint paths) after `render` and `benchmark`.

## Requirements

- Python **3.11+** (see `pyproject.toml`).

## Project layout

```text
glyphcast/                 # installable package
  commands/                # CLI: render, train-chars, benchmark, download-models
  io/                      # Video and GIF decoding
  models/                  # Edge backends and character models
  pipeline/                # Preprocess, tiling, mapping, temporal smoothing
  render/                  # Terminal, text export, video overlay
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

4. Download edge checkpoints (optional; enables DexiNed/HED when `[edge]` + PyTorch are available):

```bash
glyphcast download-models --edge all
```

Character CNN weights live under `artifacts/models/chars/` by default; train your own with `train-chars` or supply paths via config.

## Configuration

- **Presets** (`--preset`): `default`, `fast`, `high_quality` — YAML under `configs/`.
- **Charset** (`train-chars --charset` or `runtime.charset` in YAML): `minimal`, `balanced`, or `dense` (distinct progressive charsets), or a custom string.

## Commands

| Command | Purpose |
|--------|---------|
| `glyphcast render <input> --mode {terminal,text,video} [--output PATH] [--preset NAME]` | Run full pipeline on a video or GIF. |
| `glyphcast benchmark <input> [--preset NAME]` | One-frame timing and metadata; prints runtime summary. |
| `glyphcast train-chars [--charset minimal\|balanced\|dense] [--fonts PATH] [--preset NAME]` | Train the char CNN. |
| `glyphcast download-models [--edge dexined\|hed\|all] [--destination DIR]` | Fetch edge weights into `artifacts/models/edge/` (default). |

Use `glyphcast <command> --help` for full options.

## Notes

- Without `[edge]` or without downloaded edge weights, the pipeline falls back to Sobel while keeping the same CLI and config fields.
- `render` and `benchmark` echo a short runtime summary so you can confirm GPU vs CPU, edge backend, and glyph mode for that run.
- For neural edges, install `[edge]` and run `download-models`; char inference still uses `artifacts/models/chars/` when `glyph_mode` expects a CNN.

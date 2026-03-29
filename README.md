# glyphcast

GPU-accelerated ML pipeline that converts videos and GIFs into clean, noise-free ASCII art using edge detection and character classification.

## Features

- PyTorch-ready edge detection pipeline with `dexined`, `hed`, and `sobel` backends.
- Synthetic glyph dataset generation and a lightweight CNN for ASCII character classification.
- OpenCV-based video ingestion plus GIF decoding support.
- Motion-aware temporal smoothing to reduce flicker between frames.
- Output modes for terminal playback, text export, and video overlay rendering.
- Typer-based CLI for rendering, benchmarking, training, and model setup.

## Project Layout

```text
glyphcast/
  commands/      CLI commands
  io/            Video and GIF decoding
  models/        Edge backends and character models
  pipeline/      Frame preprocessing, tiling, mapping, smoothing
  render/        Terminal, text, and video output
  training/      Synthetic glyph dataset and training scripts
  utils/         Profiling utilities
configs/         Default, fast, and high-quality presets
scripts/         Helper scripts for edge model setup
tests/           End-to-end and unit coverage
```

## Quick Start

1. Create a Python `3.11` environment.
2. Install the package:

```bash
python3 -m pip install -e ".[dev,train,edge]"
```

3. Render a GIF to text:

```bash
python3 -m glyphcast.cli render giphy.gif --mode text --output artifacts/renders/giphy.txt
```

4. Prepare edge model directories:

```bash
python3 -m glyphcast.cli download-models --edge all
```

## Commands

- `python3 -m glyphcast.cli render <input> --mode {terminal,text,video}`
- `python3 -m glyphcast.cli train-chars --charset balanced`
- `python3 -m glyphcast.cli benchmark <input>`
- `python3 -m glyphcast.cli download-models --edge {dexined,hed,all}`

## Notes

- The current edge wrappers preserve the production API for DexiNed and HED while falling back to a Sobel implementation until pretrained checkpoints are downloaded.
- The frame pipeline uses template scoring for inference-time character mapping and includes a trainable CNN plus Random Forest baseline for offline model development.

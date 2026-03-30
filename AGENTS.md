## Learned User Preferences

- For glyphcast, treat "AI" as local learned inference (GPU edge and glyph models), not LLM features, unless the user says otherwise.
- Prefer knowing whether a run used GPU/ML versus CPU or fallback backends; runtime summaries matter more than assuming config alone.
- When planning GPU work, assume NVIDIA CUDA as the primary target with a usable CPU fallback path.

## Learned Workspace Facts

- The package requires Python 3.11+ per `pyproject.toml`; older system Pythons are not supported without upgrading.
- `render` and `benchmark` can print runtime summaries (device, edge backend, glyph mode, checkpoint paths) so you can confirm what actually ran.
- Char CNN inference uses checkpoints under `artifacts/models/chars/` by default; edge may stay on Sobel until optional `[edge]` dependencies and downloaded edge weights are present.
- Charset presets `minimal`, `balanced`, and `dense` must resolve to different progressively longer charsets via `CHARSET_PRESETS` in `glyphcast/constants.py`, not one shared minimal string.
- ASCII video blending resizes the source frame to the overlay size when resolutions differ so compositing does not shape-mismatch.
- `download-models` installs real edge artifacts under `artifacts/models/`; `giphy.gif` is a typical smoke-test input.

## Tooling

Run these commands locally to verify changes:

```bash
# Install development dependencies
python3 -m pip install -e ".[dev]"

# Run tests
pytest

# Lint and format check
ruff check .
ruff format --check .

# Type checking
make typecheck  # runs mypy glyphcast
mypy glyphcast
```

Notes:
- Full type checking with mypy may require `pip install -e ".[dev,train,edge]"` for optional dependencies (kornia, sklearn, joblib).
- CUDA is the primary target GPU backend; CPU fallback works when CUDA is unavailable.
- Without `[edge]` or downloaded edge weights, the pipeline falls back to Sobel while keeping the same CLI and config fields.

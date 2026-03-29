from pathlib import Path

from glyphcast.config import GlyphcastConfig, load_config
from glyphcast.constants import CHARSET_PRESETS, MINIMAL_CHARSET


def test_load_default_config_uses_expected_gpu_friendly_defaults() -> None:
    config = load_config(Path("configs/default.yaml"))

    assert config.runtime.device == "cuda"
    assert config.runtime.edge_backend == "dexined"
    assert config.runtime.glyph_mode == "cnn_plus_template"
    assert config.runtime.edge_checkpoint == "artifacts/models/edge/dexined.pt"
    assert config.runtime.char_model_path == "artifacts/models/chars/char_cnn.pt"
    assert config.render.columns == 120
    assert config.training.cell_height == 12


def test_from_preset_exposes_fast_profile() -> None:
    config = GlyphcastConfig.from_preset("fast")

    assert config.render.columns < 120
    assert config.runtime.charset == "minimal"
    assert config.runtime.batch_size == 1024


def test_render_style_charset_resolution_maps_presets_to_distinct_lengths() -> None:
    """Regression: render_command must use CHARSET_PRESETS, not one charset for all."""

    def resolved_charset(preset: str) -> str:
        cfg = GlyphcastConfig.from_preset(preset)
        return CHARSET_PRESETS.get(cfg.runtime.charset, MINIMAL_CHARSET)

    minimal = resolved_charset("fast")
    balanced = resolved_charset("default")
    dense = resolved_charset("high_quality")

    assert minimal == CHARSET_PRESETS["minimal"]
    assert balanced == CHARSET_PRESETS["balanced"]
    assert dense == CHARSET_PRESETS["dense"]
    assert len(minimal) < len(balanced) < len(dense)

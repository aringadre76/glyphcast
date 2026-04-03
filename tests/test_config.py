from pathlib import Path

from glyphcast.config import GlyphcastConfig, load_config
from glyphcast.constants import BALANCED_CHARSET, CHARSET_PRESETS, DENSE_CHARSET, MINIMAL_CHARSET


def test_load_default_config_uses_expected_defaults() -> None:
    config = load_config(Path("configs/default.yaml"))

    assert config.runtime.device == "cuda"
    assert config.runtime.edge_backend == "dexined"
    assert config.runtime.glyph_mode == "template"
    assert config.runtime.edge_checkpoint == "artifacts/models/edge/dexined.pt"
    assert config.runtime.char_model_path == "None"
    assert config.runtime.charset == "minimal"
    assert config.runtime.background_suppression is False
    assert config.render.columns == 120
    assert config.training.cell_height == 12


def test_from_preset_exposes_fast_profile() -> None:
    config = GlyphcastConfig.from_preset("fast")

    assert config.render.columns < 120
    assert config.runtime.charset == "minimal"
    assert config.runtime.batch_size == 1024
    assert config.runtime.glyph_mode == "luminance"
    assert config.runtime.background_suppression is True
    assert config.runtime.background_edge_threshold == 0.02
    assert config.runtime.background_variance_threshold == 0.0001
    assert config.runtime.background_confidence_margin == 0.1


def test_render_style_charset_resolution_uses_expected_preset_mapping() -> None:
    """Regression: render_command must use the configured charset preset per profile."""

    def resolved_charset(preset: str) -> str:
        cfg = GlyphcastConfig.from_preset(preset)
        return CHARSET_PRESETS.get(cfg.runtime.charset, MINIMAL_CHARSET)

    fast = resolved_charset("fast")
    balanced = resolved_charset("default")
    dense = resolved_charset("high_quality")

    assert fast == CHARSET_PRESETS["minimal"]
    assert balanced == CHARSET_PRESETS["minimal"]
    assert dense == CHARSET_PRESETS["dense"]
    # Charset presets must be strictly increasing in length
    assert len(MINIMAL_CHARSET) < len(BALANCED_CHARSET) < len(DENSE_CHARSET)

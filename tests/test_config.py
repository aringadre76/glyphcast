from pathlib import Path

from glyphcast.config import GlyphcastConfig, load_config


def test_load_default_config_uses_expected_gpu_friendly_defaults() -> None:
    config = load_config(Path("configs/default.yaml"))

    assert config.runtime.device == "cuda"
    assert config.runtime.edge_backend == "dexined"
    assert config.render.columns == 120
    assert config.training.cell_height == 12


def test_from_preset_exposes_fast_profile() -> None:
    config = GlyphcastConfig.from_preset("fast")

    assert config.render.columns < 120
    assert config.runtime.charset == "minimal"
    assert config.runtime.batch_size == 1024

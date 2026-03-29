"""Project-wide constants."""

from pathlib import Path

MINIMAL_CHARSET = " .:-=+*#%@"
BALANCED_CHARSET = " .'`^\",:;Il!i~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
DENSE_CHARSET = " .'`^\",:;Il!i~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$▒▓█"

CHARSET_PRESETS = {
    "minimal": MINIMAL_CHARSET,
    "balanced": BALANCED_CHARSET,
    "dense": DENSE_CHARSET,
}

DEFAULT_FONT_NAME = "DejaVuSansMono.ttf"
DEFAULT_CONFIG_DIR = Path("configs")
DEFAULT_ARTIFACT_DIR = Path("artifacts")
EDGE_MODEL_DIR = DEFAULT_ARTIFACT_DIR / "models" / "edge"
CHAR_MODEL_DIR = DEFAULT_ARTIFACT_DIR / "models" / "chars"
CACHE_DIR = DEFAULT_ARTIFACT_DIR / "cache"
RENDER_DIR = DEFAULT_ARTIFACT_DIR / "renders"

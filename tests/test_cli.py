import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from typer.testing import CliRunner

from glyphcast.cli import app


runner = CliRunner()


def test_cli_lists_core_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "render" in result.output
    assert "train-chars" in result.output
    assert "download-models" in result.output


def test_benchmark_command_accepts_input_argument() -> None:
    result = runner.invoke(app, ["benchmark", "sample.mp4"])

    assert result.exit_code == 0
    assert "sample.mp4" in result.output


def test_render_text_mode_writes_output_file(tmp_path: Path) -> None:
    gif_path = tmp_path / "sample.gif"
    output_path = tmp_path / "ascii.txt"
    Image.new("RGB", (8, 12), color=(255, 255, 255)).save(gif_path, save_all=True)

    result = runner.invoke(
        app,
        ["render", str(gif_path), "--mode", "text", "--output", str(output_path)],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert len(output_path.read_text(encoding="utf-8")) > 0


def test_render_threads_runtime_settings_into_frame_pipeline(tmp_path: Path) -> None:
    gif_path = tmp_path / "sample.gif"
    output_path = tmp_path / "ascii.txt"
    Image.new("RGB", (8, 12), color=(255, 255, 255)).save(gif_path, save_all=True)

    captured: dict[str, object] = {}

    class FakeAsciiFrame:
        width = 1
        height = 1

        def as_text(self) -> str:
            return "@"

    class FakeArtifacts:
        ascii_frame = FakeAsciiFrame()
        logits = None
        edge_maps = type("EdgeMaps", (), {"binary": None})()

    class FakePipeline:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def process_frame(self, frame_bgr):
            return FakeArtifacts()

        def runtime_summary(self) -> dict[str, str]:
            return {
                "device": "cuda",
                "edge_backend": "dexined",
                "edge_device": "cuda",
                "edge_checkpoint": "artifacts/models/edge/dexined.pt",
                "glyph_mode": "cnn_plus_template",
                "glyph_device": "cuda",
                "char_model_path": "artifacts/models/chars/char_cnn.pt",
            }

    with patch("glyphcast.commands.render.FramePipeline", FakePipeline):
        result = runner.invoke(
            app,
            ["render", str(gif_path), "--mode", "text", "--output", str(output_path)],
        )

    assert result.exit_code == 0
    assert captured["device"] == "cuda"
    assert captured["mixed_precision"] is True
    assert captured["batch_size"] == 512
    assert captured["glyph_mode"] == "cnn_plus_template"
    assert captured["edge_checkpoint"] == "artifacts/models/edge/dexined.pt"
    assert captured["char_model_path"] == "artifacts/models/chars/char_cnn.pt"
    assert "device=cuda" in result.output


def test_download_models_command_creates_destination(tmp_path: Path) -> None:
    destination = tmp_path / "edge-models"

    result = runner.invoke(
        app,
        ["download-models", "--destination", str(destination)],
    )

    assert result.exit_code == 0
    assert destination.exists()


def test_download_models_command_writes_checkpoints_and_manifest(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "edge-models"

    class FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return self.payload

    monkeypatch.setattr(
        "glyphcast.commands.models.urlopen",
        lambda _url: FakeResponse(b"checkpoint-bytes"),
    )

    result = runner.invoke(
        app,
        ["download-models", "--edge", "all", "--destination", str(destination)],
    )

    assert result.exit_code == 0
    assert (destination / "dexined.pt").read_bytes() == b"checkpoint-bytes"
    assert (destination / "hed.pth").read_bytes() == b"checkpoint-bytes"

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))

    assert sorted(manifest["models"]) == ["dexined", "hed"]
    assert manifest["models"]["dexined"]["path"] == "dexined.pt"
    assert manifest["models"]["hed"]["path"] == "hed.pth"


def test_train_chars_threads_runtime_settings_into_training_command() -> None:
    captured: dict[str, object] = {}

    def fake_train_char_cnn(charset: str, **kwargs: object) -> Path:
        captured["charset"] = charset
        captured.update(kwargs)
        return Path("artifacts/models/chars/char_cnn.pt")

    with patch("glyphcast.commands.train.train_char_cnn", fake_train_char_cnn):
        result = runner.invoke(app, ["train-chars", "--preset", "fast"])

    assert result.exit_code == 0
    assert captured["charset"] == " .:-=+*#%@"
    assert captured["device"] == "cuda"
    assert captured["cell_size"] == (8, 12)
    assert captured["fonts"] == ["DejaVuSansMono.ttf"]

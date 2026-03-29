from pathlib import Path

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


def test_download_models_command_creates_destination(tmp_path: Path) -> None:
    destination = tmp_path / "edge-models"

    result = runner.invoke(
        app,
        ["download-models", "--destination", str(destination)],
    )

    assert result.exit_code == 0
    assert destination.exists()

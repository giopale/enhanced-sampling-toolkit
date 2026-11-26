"""Figures shown in the terminal (--show)."""

import base64
import io
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from estk.cli import show  # noqa: E402
from estk.cli.main import _state, app  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\nfake"


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_inline_sequence():
    text = show.inline_sequence(PNG)
    assert text.startswith("\033]1337;File=") and text.endswith("\a\n")
    assert f"size={len(PNG)};inline=1" in text
    assert text.split(":", 1)[1].rstrip("\a\n") == base64.b64encode(PNG).decode()
    wrapped = show.inline_sequence(PNG, tmux=True)
    assert wrapped.startswith("\033Ptmux;\033\033]1337;") and wrapped.endswith("\033\\\n")


def test_show_png(monkeypatch, tmp_path):
    assert show.show_png(PNG, stream=io.StringIO()) is False  # not a terminal: nothing
    monkeypatch.delenv("ESTK_IMGCAT", raising=False)
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(show.shutil, "which", lambda _: None)
    out = Terminal()
    assert show.show_png(PNG, stream=out)
    assert out.getvalue() == show.inline_sequence(PNG)

    log = tmp_path / "calls"
    viewer = tmp_path / "viewer"
    viewer.write_text(  # copies its last argument, the image
        f'#!/bin/sh\nfor f; do :; done\ncp "$f" {tmp_path}/shown.png\necho "$@" >> {log}\n'
    )
    viewer.chmod(0o755)
    monkeypatch.setenv("ESTK_IMGCAT", f"{viewer} --flag")
    out = Terminal()
    assert show.show_png(PNG, stream=out)
    assert out.getvalue() == ""  # the viewer showed it
    assert (tmp_path / "shown.png").read_bytes() == PNG
    assert log.read_text().startswith("--flag ")

    monkeypatch.setenv("ESTK_IMGCAT", "false")  # a viewer that fails: inline image instead
    out = Terminal()
    assert show.show_png(PNG, stream=out) and "1337;File=" in out.getvalue()


def test_show_figure(monkeypatch):
    shown = []
    monkeypatch.setattr(show, "show_png", lambda data: shown.append(data) or True)
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    assert show.show_figure(fig) and shown[0].startswith(b"\x89PNG")


def test_show_anywhere_on_the_command_line(monkeypatch):
    assert show.move_show_first(["estk", "plot", "surface", "f.dat", "--show"]) == [
        "estk", "--show", "plot", "surface", "f.dat"]  # fmt: skip
    assert show.move_show_first(["estk", "plot"]) == ["estk", "plot"]
    calls = []
    monkeypatch.setattr("estk.cli.main.app", lambda: calls.append(list(sys.argv)))
    monkeypatch.setattr(sys, "argv", ["estk", "info", "--show"])
    from estk.cli.main import run

    run()
    assert calls == [["estk", "--show", "info"]]


def test_command(monkeypatch, tmp_path):
    from estk.grid import Grid
    from estk.surface import Surface

    grid = Grid(("x", "y"), (0.0, 0.0), (1.0, 1.0), (11, 11))
    values = [[(i - 5) ** 2 + (j - 5) ** 2 for j in range(11)] for i in range(11)]
    Surface(grid, values, metadata={"kbt": 2.5}).save(tmp_path / "s.dat")
    shown = []
    monkeypatch.setattr(show, "show_figure", lambda fig: shown.append(fig) or True)
    runner = CliRunner()
    result = runner.invoke(app, ["--show", "plot", "surface", str(tmp_path / "s.dat"),
                                 "-o", str(tmp_path / "s.png")])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert len(shown) == 1 and (tmp_path / "s.png").is_file()
    result = runner.invoke(app, ["plot", "surface", str(tmp_path / "s.dat"),
                                 "-o", str(tmp_path / "t.png")])  # fmt: skip
    assert result.exit_code == 0 and len(shown) == 1  # without --show: not shown
    assert _state["show"] is False


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    _state["show"] = False

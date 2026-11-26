"""Figures shown in the terminal (``--show``).

With an ``imgcat`` program on the PATH (iTerm2's, or any that takes image files), or the command
in ``ESTK_IMGCAT`` (e.g. ``kitten icat`` for kitty), estk runs it on a PNG of the figure.
Otherwise it writes the inline-image escape sequence itself (the iTerm2 protocol, also read by
WezTerm and others), wrapped for tmux when inside it (tmux needs ``set -g
allow-passthrough on``). Nothing is shown when the output is not a terminal.
"""

from __future__ import annotations

import base64
import io
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import IO, Any

from estk.logging import get_logger

log = get_logger(__name__)


def show_figure(fig: Any) -> bool:
    """Show a matplotlib figure in the terminal; True if something was shown."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    return show_png(buffer.getvalue())


def show_png(data: bytes, stream: IO[str] | None = None) -> bool:
    stream = stream or sys.stdout
    if not stream.isatty():
        log.warning("--show: the output is not a terminal, so the figure is not shown")
        return False
    command = os.environ.get("ESTK_IMGCAT") or shutil.which("imgcat")
    if command:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "estk.png"
            path.write_bytes(data)
            stream.flush()
            done = subprocess.run([*shlex.split(command), str(path)], check=False)
        if done.returncode == 0:
            return True
        log.warning("--show: %s failed; writing the inline image directly", command)
    stream.write(inline_sequence(data, tmux=bool(os.environ.get("TMUX"))))
    stream.flush()
    return True


def inline_sequence(data: bytes, *, tmux: bool = False, name: str = "estk.png") -> str:
    """The iTerm2 inline-image escape sequence for a PNG (wrapped for tmux passthrough)."""
    encoded_name = base64.b64encode(name.encode()).decode()
    payload = base64.b64encode(data).decode()
    sequence = (f"\033]1337;File=name={encoded_name};size={len(data)};inline=1;"
                f"preserveAspectRatio=1:{payload}\a")  # fmt: skip
    if tmux:
        sequence = "\033Ptmux;" + sequence.replace("\033", "\033\033") + "\033\\"
    return sequence + "\n"


def move_show_first(argv: list[str]) -> list[str]:
    """``estk plot surface f.dat --show`` -> ``estk --show plot surface f.dat``: --show is an
    option of estk itself, but may be written anywhere on the command line."""
    if "--show" not in argv[1:]:
        return argv
    return [argv[0], "--show", *[a for a in argv[1:] if a != "--show"]]

#!/usr/bin/env python3
"""Make a small copy of simulation runs for the regression tests.

For each run folder it copies:
  COLVAR, CV-*      the first --rows frames (after keeping every --stride-th COLVAR row);
                    CV-* files keep the same times as COLVAR
  log.lammps        without the thermo output (only the commands are needed)
  plumed.plumed,
  lammps.input      unchanged
  the state file    header only (named by STATE_WFILE in the PLUMED input)
Trajectories, restarts, kernels and other logs are not copied.

Example:
  python tools/make_test_data.py /path/to/A tests/data/runs --runs 000 001 002 --rows 3000

Only the standard library is used, so it runs in any Python >= 3.8.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

PLAIN_COPIES = ("plumed.plumed", "lammps.input")
TIME_TOLERANCE = 1e-6  # same frame if times differ by less than this


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("source", type=Path, help="folder containing the run folders")
    parser.add_argument("dest", type=Path, help="where to write the trimmed runs")
    parser.add_argument("--runs", nargs="+", required=True, help="run folder names, e.g. 000 001")
    parser.add_argument("--rows", type=int, default=3000, help="COLVAR rows to keep (default 3000)")
    parser.add_argument(
        "--stride", type=int, default=1, help="keep every n-th COLVAR row (default 1)"
    )
    parser.add_argument("--colvar", default="COLVAR", help="COLVAR file name (default COLVAR)")
    parser.add_argument("--force", action="store_true", help="overwrite existing output folders")
    args = parser.parse_args()

    for name in args.runs:
        src, dst = args.source / name, args.dest / name
        if not (src / args.colvar).is_file():
            sys.exit(f"{src / args.colvar} not found")
        if dst.exists():
            if not args.force:
                sys.exit(f"{dst} exists (use --force to overwrite)")
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        trim_run(src, dst, args)

    (args.dest / "PROVENANCE.txt").write_text(
        "Trimmed copy of simulation runs, made with:\n  "
        + " ".join(sys.argv)
        + f"\nsource: {args.source.resolve()}\n"
    )
    print(f"done; total size {total_size(args.dest) / 1e6:.2f} MB")


def trim_run(src: Path, dst: Path, args: argparse.Namespace) -> None:
    keys = trim_colvar(src / args.colvar, dst / args.colvar, args.rows, args.stride)
    report = [f"{args.colvar}: {len(keys)} times"]

    for path in sorted(src.glob("CV-*")):
        n = filter_by_time(path, dst / path.name, keys)
        report.append(f"{path.name}: {n} rows")
        if n == 0:
            print(f"  warning: {path.name} has no times in common with {args.colvar}")

    if (src / "log.lammps").is_file():
        n = strip_thermo(src / "log.lammps", dst / "log.lammps")
        report.append(f"log.lammps: {n} lines")

    for name in PLAIN_COPIES:
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)
            report.append(name)

    state = state_file_name(src / "plumed.plumed")
    if state and (src / state).is_file():
        n = copy_header(src / state, dst / state)
        report.append(f"{state}: {n} header lines")
    elif state:
        print(f"  warning: state file {state} not found in {src}")

    print(f"{src.name}: " + ", ".join(report))


def time_key(word: str) -> int:
    return round(float(word) / TIME_TOLERANCE)


def trim_colvar(src: Path, dst: Path, rows: int, stride: int) -> set:
    keys = set()
    kept = 0
    index = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if line.startswith("#"):
                if kept < rows:
                    fout.write(line)
                continue
            if not line.strip():
                continue
            if index % stride == 0:
                if kept >= rows:
                    break
                fout.write(line)
                keys.add(time_key(line.split()[0]))
                kept += 1
            index += 1
    return keys


def filter_by_time(src: Path, dst: Path, keys: set) -> int:
    n = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if line.startswith("#"):
                fout.write(line)
                continue
            words = line.split()
            if not words:
                continue
            try:
                key = time_key(words[0])
            except ValueError:
                continue
            if key in keys:
                fout.write(line)
                n += 1
    return n


_NUMBER = re.compile(r"^\s*[-+]?(\d|\.\d)")


def strip_thermo(src: Path, dst: Path) -> int:
    """Copy the log without lines that start with a number (thermo output)."""
    n = 0
    with open(src, errors="replace") as fin, open(dst, "w") as fout:
        for line in fin:
            if not _NUMBER.match(line):
                fout.write(line)
                n += 1
    return n


def state_file_name(plumed: Path):
    if not plumed.is_file():
        return None
    match = re.search(r"STATE_WFILE=(\S+)", plumed.read_text(errors="replace"))
    return match.group(1) if match else None


def copy_header(src: Path, dst: Path) -> int:
    n = 0
    with open(src, errors="replace") as fin, open(dst, "w") as fout:
        for line in fin:
            if not line.startswith("#"):
                break
            fout.write(line)
            n += 1
    return n


def total_size(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


if __name__ == "__main__":
    main()

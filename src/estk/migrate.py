"""Convert an old Hydra case file (``to_refactor/config/case/*.yaml``) to ``estk.yaml``.

Only facts about the system are carried over: run location, file names, CV limits and
bandwidths, atom types and atom index groups. Everything else (plot cosmetics, settings of
individual commands, choices for a single run) is reported as dropped, with a pointer to where
it lives in the new design. YAML anchors (``&name`` / ``*name``) of the values that are kept
are preserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from estk import _yaml
from estk.errors import ConfigError

# Where the settings that are not migrated go in the new design (sections of REFACTOR.md).
_DROPPED_HINTS = {
    "mplstyle": "plot styles, --style (§12)",
    "size": "plot styles (§12)",
    "surface_data": "estk mep / estk plot flags and plot styles (§10, §12)",
    "convergence_barrier": "estk convergence flags (§11)",
    "cvs_along_mep": "estk along-path flags (§13)",
    "find_structures": "type_map -> [atoms]; the rest -> estk structures flags (§14)",
    "find_structures_level_set": "type_map -> [atoms]; the rest -> estk structures flags (§14)",
    "proton_transfer": "atom indices -> [atoms.groups]; the rest -> estk cv proton-transfer (§15)",
    "temp": "read from log.lammps (§4)",
    "bias_factor": "read from the PLUMED state file (§4)",
    "cvs.select": "--cvs flag",
    "cvs.add": "--quantities flag (§13)",
    "cvs.filter": "--filter flag (§5)",
}
_DEFAULT_HINT = "flag of the relevant command"


def migrate_case_yaml(path: Path | str) -> tuple[str, list[str]]:
    """Return ``(yaml_text, dropped)`` for an old case YAML file.

    ``dropped`` lists the settings that were not migrated, each with a hint of where it goes now.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError("file not found", path=path)
    try:
        data, anchors = _yaml.load(path.read_text())
        data = data or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}", path=path) from None
    if not isinstance(data, dict):
        raise ConfigError("expected a mapping at the top level", path=path)

    out: dict[str, dict[str, Any]] = {"runs": {}, "files": {}, "cv": {}, "atoms": {}}
    dropped: list[str] = []
    used: set[str] = set()

    def take(key: str, section: str, name: str) -> None:
        if data.get(key) is not None:
            out[section][name] = data[key]
            used.add(key)

    take("dir_md", "runs", "root")
    take("dir_job", "runs", "pattern")
    take("file_lammps_log", "files", "lammps_log")
    take("file_plumed_input", "files", "plumed_input")
    take("file_model_devi", "files", "model_deviation")

    cvs = data.get("cvs") or {}
    used.add("cvs")
    for name, settings in cvs.items():
        if name in ("select", "add", "filter"):
            if settings is not None:
                dropped.append(_hint(f"cvs.{name}"))
            continue
        cv, rest = _migrate_cv(settings or {})
        out["cv"][str(name)] = cv
        dropped += [_hint(f"cvs.{name}.{key}") for key in rest]

    type_map = _find_type_map(data)
    if type_map:
        out["atoms"]["type_map"] = type_map

    groups = _proton_transfer_groups(data.get("proton_transfer") or {})
    if groups:
        out["atoms"]["groups"] = groups

    for key in data:
        if key not in used:
            dropped.append(_hint(key))

    header = (
        f"# Migrated from {path.name} by `estk config migrate`.\n"
        "# Settings that are not facts about the system were left out; see the list printed\n"
        "# by the command.\n\n"
    )
    out = {section: values for section, values in out.items() if values}
    body = _yaml.dump(out, anchors)
    # blank line between top-level sections
    body = "\n".join(
        ("\n" + line) if line[:1].isalpha() and i else line
        for i, line in enumerate(body.splitlines())
    )
    return header + body + "\n", dropped


def _migrate_cv(settings: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    cv: dict[str, Any] = {}
    rest = []
    for key, value in settings.items():
        if key == "limits" and value is not None:
            if any(v is not None for v in value):
                cv["limits"] = value  # same object, so a shared anchor is kept
        elif key == "sigma":
            if value is not None:
                cv["bandwidth"] = value
        elif key == "scale":
            cv["scale"] = value
        else:
            rest.append(key)
    return cv, rest


def _find_type_map(data: dict[str, Any]) -> dict[int, str]:
    for section in ("find_structures", "find_structures_level_set"):
        save = (data.get(section) or {}).get("save_structures") or {}
        if save.get("type_map"):
            return {int(k): str(v) for k, v in save["type_map"].items()}
    return {}


def _proton_transfer_groups(pt: dict[str, Any]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}

    def add(name: str, value: Any) -> None:
        if value is None:
            return
        groups[name] = value if isinstance(value, list) else [value]  # keeps anchors

    add("pt_type1", pt.get("indices_type1"))
    add("pt_type2", pt.get("indices_type2"))
    add("water_oxygens", pt.get("indices_water"))
    for site, settings in (pt.get("surf_sites") or {}).items():
        add(f"site_{site}", (settings or {}).get("indices_type2"))
    return groups


def _hint(key: str) -> str:
    hint = _DROPPED_HINTS.get(key)
    if hint is None and key.startswith("cvs."):
        hint = "plot styles or the relevant command's flags"
    return f"{key} -> {hint or _DEFAULT_HINT}"

"""YAML reading and writing used by the configuration code.

``Loader`` is PyYAML's safe loader with two additions: duplicated keys are an error (plain PyYAML
silently keeps the last one), and the names of anchors (``&name``) are remembered so that
``dump`` can write them back. ``dump`` writes mappings in block style, lists of plain values on
one line, and keeps anchors for objects that appear more than once.
"""

from __future__ import annotations

import json
import re
from typing import Any

import yaml
from yaml.constructor import ConstructorError

_PLAIN_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")
# Words that YAML 1.1 reads as booleans or null when unquoted.
_RESERVED = {"y", "n", "yes", "no", "on", "off", "true", "false", "null"}


class Loader(yaml.SafeLoader):
    def __init__(self, stream: Any) -> None:
        super().__init__(stream)
        self.anchor_names: dict[int, str] = {}  # id(constructed object) -> anchor name
        self._node_anchors: dict[int, tuple[str, yaml.Node]] = {}

    def compose_node(self, parent: Any, index: Any) -> yaml.Node:
        event = self.peek_event()
        anchor = None if isinstance(event, yaml.AliasEvent) else getattr(event, "anchor", None)
        node = super().compose_node(parent, index)
        if anchor:
            self._node_anchors[id(node)] = (anchor, node)
        return node

    def construct_object(self, node: yaml.Node, deep: bool = False) -> Any:
        obj = super().construct_object(node, deep=deep)
        if id(node) in self._node_anchors:
            self.anchor_names[id(obj)] = self._node_anchors[id(node)][0]
        return obj

    def construct_mapping(self, node: yaml.Node, deep: bool = False) -> dict[Any, Any]:
        seen: dict[Any, yaml.Node] = {}
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if key in seen:
                raise ConstructorError(
                    "while reading a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen[key] = key_node
        return super().construct_mapping(node, deep=deep)


def load(text: str) -> tuple[Any, dict[int, str]]:
    """Parse ``text``; return the data and a map from ``id(object)`` to its anchor name."""
    loader = Loader(text)
    try:
        return loader.get_single_data(), loader.anchor_names
    finally:
        loader.dispose()


class _Dumper(yaml.SafeDumper):
    anchor_names: dict[int, str] = {}

    def generate_anchor(self, node: yaml.Node) -> str:
        for obj_id, represented in self.represented_objects.items():
            if represented is node and obj_id in self.anchor_names:
                name = self.anchor_names[obj_id]
                break
        else:
            return super().generate_anchor(node)
        used = getattr(self, "_used_anchors", set())
        unique, n = name, 1
        while unique in used:
            n += 1
            unique = f"{name}_{n}"
        used.add(unique)
        self._used_anchors = used
        return unique

    def represent_list(self, data: list[Any]) -> yaml.Node:
        flow = all(not isinstance(v, (dict, list)) for v in data)
        return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _Dumper.represent_list)


def dump(data: Any, anchor_names: dict[int, str] | None = None) -> str:
    """Write ``data`` as block-style YAML, keeping the given anchor names for shared objects."""
    dumper = type("Dumper", (_Dumper,), {"anchor_names": dict(anchor_names or {})})
    return yaml.dump(data, Dumper=dumper, sort_keys=False, allow_unicode=True, width=100, indent=2)


def scalar(value: str) -> str:
    """Quote a string for YAML."""
    return json.dumps(value, ensure_ascii=False)


def key(name: str) -> str:
    """Quote a mapping key only when needed."""
    plain = _PLAIN_KEY.match(name) and name.lower() not in _RESERVED
    return name if plain else scalar(name)

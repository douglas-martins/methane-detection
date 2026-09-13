"""Make mutmut's path-derived names compatible with this repository's flat imports.

The plugin is loaded only by mutmut's internal pytest runs. It leaves normal
pytest and runtime imports unchanged.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

_PATCH_MARKER = "_methane_detection_flat_import_compatibility"
_ORIGINAL_RECORD_ATTR = "_methane_detection_original_record_trampoline_hit"
_ORIGINAL_WRAP_ATTR = "_methane_detection_original_wrap_in_trampoline"
_MUTMUT_FUNCTION_PREFIXES = ("x_", "xǁ")


def _module_name_from_source_path(source_path: Path, *, root: Path) -> str | None:
    """Return mutmut's dotted module name for a file under ``src`` or ``flows``."""
    try:
        relative_path = source_path.resolve().relative_to(root.resolve())
    except ValueError:
        return None

    if relative_path.suffix != ".py" or not relative_path.parts:
        return None

    parts = list(relative_path.with_suffix("").parts)
    if parts[0] == "src":
        parts.pop(0)
    elif parts[0] != "flows":
        return None

    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or None


def _module_name_from_mutmut_key(key: str) -> str | None:
    """Extract the imported module portion from a mangled mutmut function key."""
    parts = key.split(".")
    for index, part in enumerate(parts):
        if part.startswith(_MUTMUT_FUNCTION_PREFIXES):
            return ".".join(parts[:index]) or None
    return None


def _qualify_mutmut_key(
    key: str | None,
    *,
    root: Path,
    modules: Mapping[str, ModuleType | Any],
) -> str | None:
    """Replace a flat module prefix with mutmut's source-path-derived prefix."""
    if key is None:
        return None

    imported_name = _module_name_from_mutmut_key(key)
    if imported_name is None:
        return key

    module = modules.get(imported_name)
    source_file = getattr(module, "__file__", None)
    if source_file is None:
        return key

    qualified_name = _module_name_from_source_path(Path(source_file), root=root)
    if qualified_name is None or qualified_name == imported_name:
        return key
    return qualified_name + key[len(imported_name) :]


def _install_flat_import_compatibility(*, root: Path) -> None:
    """Patch mutmut's trampoline naming while preserving ordinary flat imports."""
    from mutmut.mutation import trampoline

    if getattr(trampoline.wrap_in_trampoline, _PATCH_MARKER, False):
        return

    original_wrap_in_trampoline = trampoline.wrap_in_trampoline
    original_record_trampoline_hit = trampoline.record_trampoline_hit

    def compatible_wrap_in_trampoline(
        mutants_dict: dict[str, Any], is_classmethod: bool = False
    ) -> Any:
        def decorator(function: Any) -> Any:
            source_file = Path(function.__code__.co_filename)
            qualified_name = _module_name_from_source_path(source_file, root=root)
            if qualified_name is not None:
                function.__module__ = qualified_name
            return original_wrap_in_trampoline(mutants_dict, is_classmethod=is_classmethod)(
                function
            )

        return decorator

    def compatible_record_trampoline_hit(name: str, caller: str | None = None) -> None:
        qualified_name = _qualify_mutmut_key(name, root=root, modules=sys.modules)
        qualified_caller = _qualify_mutmut_key(caller, root=root, modules=sys.modules)
        original_record_trampoline_hit(qualified_name, caller=qualified_caller)

    setattr(compatible_wrap_in_trampoline, _PATCH_MARKER, True)
    setattr(compatible_wrap_in_trampoline, _ORIGINAL_WRAP_ATTR, original_wrap_in_trampoline)
    setattr(
        compatible_record_trampoline_hit,
        _ORIGINAL_RECORD_ATTR,
        original_record_trampoline_hit,
    )
    trampoline.wrap_in_trampoline = compatible_wrap_in_trampoline
    trampoline.record_trampoline_hit = compatible_record_trampoline_hit


def _remove_flat_import_compatibility() -> bool:
    """Restore mutmut's original trampoline functions if the plugin patched them."""
    from mutmut.mutation import trampoline

    compatible_wrap = trampoline.wrap_in_trampoline
    if not getattr(compatible_wrap, _PATCH_MARKER, False):
        return False

    trampoline.wrap_in_trampoline = getattr(compatible_wrap, _ORIGINAL_WRAP_ATTR)
    trampoline.record_trampoline_hit = getattr(
        trampoline.record_trampoline_hit,
        _ORIGINAL_RECORD_ATTR,
    )
    return True


def pytest_configure(config: Any) -> None:
    """Install flat-import compatibility before pytest collects source modules."""
    _install_flat_import_compatibility(root=Path.cwd())

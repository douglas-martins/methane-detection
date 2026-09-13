"""Tests for the mutmut compatibility plugin used with flat source imports."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import _mutmut_pytest_plugin as plugin


def test_module_name_from_source_path_removes_src_root(tmp_path):
    source_file = tmp_path / "src" / "registry" / "promotion_criteria.py"

    module_name = plugin._module_name_from_source_path(source_file, root=tmp_path)

    assert module_name == "registry.promotion_criteria"


def test_module_name_from_source_path_keeps_flows_root(tmp_path):
    source_file = tmp_path / "flows" / "retrain.py"

    module_name = plugin._module_name_from_source_path(source_file, root=tmp_path)

    assert module_name == "flows.retrain"


def test_module_name_from_source_path_ignores_files_outside_mutated_roots(tmp_path):
    source_file = tmp_path / "tests" / "helper.py"

    module_name = plugin._module_name_from_source_path(source_file, root=tmp_path)

    assert module_name is None


def test_qualify_mutmut_key_uses_loaded_flat_modules_file_path(tmp_path):
    module = ModuleType("promotion_criteria")
    module.__file__ = str(tmp_path / "src" / "registry" / "promotion_criteria.py")

    qualified = plugin._qualify_mutmut_key(
        "promotion_criteria.x_check_thresholds",
        root=tmp_path,
        modules={"promotion_criteria": module},
    )

    assert qualified == "registry.promotion_criteria.x_check_thresholds"


def test_configure_replaces_union_selection_with_active_environment_paths(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "MUTMUT_TEST_PATHS",
        "src/data/download/__tests__ tests/vendor_starcop",
    )
    monkeypatch.setattr(plugin, "_install_flat_import_compatibility", lambda root: None)
    config = SimpleNamespace(rootpath=tmp_path, args=["src", "tests"])

    plugin.pytest_configure(config)

    assert config.args == ["src/data/download/__tests__", "tests/vendor_starcop"]


def test_configure_keeps_mutmut_selected_node_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("MUTMUT_TEST_PATHS", "src/data/download/__tests__")
    monkeypatch.setattr(plugin, "_install_flat_import_compatibility", lambda root: None)
    node_id = "src/data/download/__tests__/test_download_mini_dataset.py::test_name"
    config = SimpleNamespace(rootpath=tmp_path, args=[node_id])

    plugin.pytest_configure(config)

    assert config.args == [node_id]


def test_configure_keeps_union_selection_without_environment_override(monkeypatch, tmp_path):
    monkeypatch.delenv("MUTMUT_TEST_PATHS", raising=False)
    monkeypatch.setattr(plugin, "_install_flat_import_compatibility", lambda root: None)
    config = SimpleNamespace(rootpath=tmp_path, args=["src", "tests"])

    plugin.pytest_configure(config)

    assert config.args == ["src", "tests"]


def test_installed_plugin_records_qualified_key_and_dispatches_qualified_mutant(
    tmp_path, monkeypatch
):
    from mutmut.mutation import trampoline

    source_file = tmp_path / "src" / "registry" / "promotion_criteria.py"
    source_file.parent.mkdir(parents=True)
    recorded = []
    plugin_was_installed = plugin._remove_flat_import_compatibility()
    original_record_trampoline_hit = trampoline.record_trampoline_hit
    trampoline.record_trampoline_hit = lambda name, caller=None: recorded.append((name, caller))

    try:
        plugin._install_flat_import_compatibility(root=tmp_path)

        module = ModuleType("promotion_criteria")
        module.__file__ = str(source_file)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        module.__dict__["_mutmut_mutated"] = trampoline.wrap_in_trampoline
        exec(
            compile(
                """
mutants_x_value__mutmut = {}

@_mutmut_mutated(mutants_x_value__mutmut)
def value():
    pass

def x_value__mutmut_orig():
    return "original"

def x_value__mutmut_1():
    return "mutated"

mutants_x_value__mutmut["_mutmut_orig"] = x_value__mutmut_orig
mutants_x_value__mutmut["x_value__mutmut_1"] = x_value__mutmut_1
""",
                str(source_file),
                "exec",
            ),
            module.__dict__,
        )

        trampoline.set_mutant_under_test("stats")
        assert module.value() == "original"
        assert recorded == [("registry.promotion_criteria.x_value", None)]

        trampoline.set_mutant_under_test("registry.promotion_criteria.x_value__mutmut_1")
        assert module.value() == "mutated"
    finally:
        trampoline.set_mutant_under_test(None)
        plugin._remove_flat_import_compatibility()
        trampoline.record_trampoline_hit = original_record_trampoline_hit
        if plugin_was_installed:
            plugin._install_flat_import_compatibility(root=Path.cwd())

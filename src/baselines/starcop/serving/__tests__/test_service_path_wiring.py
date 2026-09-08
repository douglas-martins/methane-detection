"""Import-path regression tests for the relocated BentoML service."""

import runpy
import sys

import service


def test_reload_restores_baseline_and_shared_serving_paths(monkeypatch):
    expected_paths = {str(service._THIS_DIR), str(service._SHARED_SERVING_DIR)}
    monkeypatch.setattr(sys, "path", [path for path in sys.path if path not in expected_paths])

    runpy.run_path(service.__file__, run_name="_service_path_wiring_test")

    assert expected_paths.issubset(sys.path)

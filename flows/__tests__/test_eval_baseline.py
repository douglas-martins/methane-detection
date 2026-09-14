"""Tests for flows/eval_baseline.py -- Phase 4's repeatable, auditable
paper-eval flow. All side-effecting steps (subprocess calls, HTTP calls,
MLflow SDK calls) are injected as fakes here, same convention as
test_retrain.py (Test Size: Small) -- this suite never touches a real
subprocess, network, or MLflow server. See
track-a-paper-benchmark-reproduction-plan.md Phase 4.
"""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import eval_baseline
import pytest


class FakeCompletedProcess(SimpleNamespace):
    returncode: int
    stdout: str = ""
    stderr: str = ""


class FakeResponse(SimpleNamespace):
    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeProcess:
    def __init__(self):
        self.terminated = False
        self.waited = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True


class TestConstants:
    def test_variants_covers_all_three(self):
        assert set(eval_baseline.VARIANTS) == {"varon", "mag1c_only", "mag1c_rgb"}

    def test_servable_variants_excludes_varon(self):
        assert "varon" not in eval_baseline.SERVABLE_VARIANTS
        assert set(eval_baseline.SERVABLE_VARIANTS) == {"mag1c_only", "mag1c_rgb"}

    def test_mlflow_tracking_uri_matches_retrain_pys_hardcoded_value(self):
        # Same public, non-secret endpoint -- retrain.py already establishes
        # not depending on MLFLOW_TRACKING_URI being present in the worker's
        # own environment.
        assert (
            eval_baseline.MLFLOW_TRACKING_URI == "https://methane-detection-mlflow.ghostface.tech"
        )


class TestMultistarcopRegisteredAndCurrent:
    def test_true_when_staging_versions_tag_matches_the_resolved_checkpoint(self):
        captured = {}
        fake_client = object()

        def fake_resolve_checkpoint(variant, dest_dir):
            captured["checkpoint"] = (variant, dest_dir)
            return (Path("ckpt"), Path("cfg"), {"checkpoint_sha256": "abc123"})

        def fake_resolve_stage_version(client, model_name, stage):
            captured["stage"] = (client, model_name, stage)
            return SimpleNamespace(run_id="run-1")

        class FakeClient:
            def get_run(self, run_id):
                captured["run_id"] = run_id
                return SimpleNamespace(data=SimpleNamespace(tags={"checkpoint_sha256": "abc123"}))

        fake_client = FakeClient()
        result = eval_baseline.multistarcop_registered_and_current(
            fake_client,
            resolve_checkpoint_fn=fake_resolve_checkpoint,
            resolve_stage_version_fn=fake_resolve_stage_version,
        )

        assert result is True
        assert captured["checkpoint"][0] == "varon"
        assert captured["checkpoint"][1].is_absolute()
        assert captured["stage"] == (fake_client, "starcop-baseline-varon", "Staging")
        assert captured["run_id"] == "run-1"

    def test_false_when_the_tag_does_not_match(self):
        def fake_resolve_checkpoint(variant, dest_dir):
            return (Path("ckpt"), Path("cfg"), {"checkpoint_sha256": "new-digest"})

        def fake_resolve_stage_version(client, model_name, stage):
            return SimpleNamespace(run_id="run-1")

        class FakeClient:
            def get_run(self, run_id):
                return SimpleNamespace(
                    data=SimpleNamespace(tags={"checkpoint_sha256": "stale-digest"})
                )

        result = eval_baseline.multistarcop_registered_and_current(
            FakeClient(),
            resolve_checkpoint_fn=fake_resolve_checkpoint,
            resolve_stage_version_fn=fake_resolve_stage_version,
        )

        assert result is False

    def test_false_when_nothing_is_registered_yet(self):
        def fake_resolve_checkpoint(variant, dest_dir):
            return (Path("ckpt"), Path("cfg"), {"checkpoint_sha256": "abc123"})

        def fake_resolve_stage_version(client, model_name, stage):
            raise ValueError(f"no version of {model_name!r} is currently at stage {stage!r}")

        result = eval_baseline.multistarcop_registered_and_current(
            object(),
            resolve_checkpoint_fn=fake_resolve_checkpoint,
            resolve_stage_version_fn=fake_resolve_stage_version,
        )

        assert result is False

    def test_checks_the_varon_model_name_specifically(self):
        received = []

        def fake_resolve_checkpoint(variant, dest_dir):
            received.append(("checkpoint", variant))
            return (Path("ckpt"), Path("cfg"), {"checkpoint_sha256": "abc123"})

        def fake_resolve_stage_version(client, model_name, stage):
            received.append(("registry", model_name, stage))
            raise ValueError("not registered")

        eval_baseline.multistarcop_registered_and_current(
            object(),
            resolve_checkpoint_fn=fake_resolve_checkpoint,
            resolve_stage_version_fn=fake_resolve_stage_version,
        )

        assert ("checkpoint", "varon") in received
        assert ("registry", "starcop-baseline-varon", "Staging") in received


class TestEnsureMultistarcopRegistered:
    def test_skips_import_when_already_registered(self):
        import_calls = []
        fake_client = object()

        def is_registered(client):
            assert client is fake_client
            return True

        eval_baseline.ensure_multistarcop_registered(
            fake_client,
            is_registered_fn=is_registered,
            import_fn=lambda variant, stage: import_calls.append((variant, stage)),
        )

        assert import_calls == []

    def test_imports_when_not_registered(self):
        import_calls = []

        eval_baseline.ensure_multistarcop_registered(
            object(),
            is_registered_fn=lambda client: False,
            import_fn=lambda variant, stage: import_calls.append((variant, stage)),
        )

        assert import_calls == [("varon", "Staging")]


class TestRunEvaluationForVariant:
    def test_returns_parsed_run_id_on_success(self):
        def fake_runner(cmd, **kwargs):
            return FakeCompletedProcess(returncode=0, stdout="MLFLOW_RUN_ID=abc123\n")

        run_id = eval_baseline.run_evaluation_for_variant(
            Path("/repo"), "mag1c_rgb", Path("/staging"), cmd_runner=fake_runner
        )

        assert run_id == "abc123"

    def test_invokes_environment_as_python_not_root_venv(self):
        # Phase 0 pinned the real 342-scene evaluation to baseline env
        # (vendor/starcop/.venv, torch 1.13.1) -- the flow process itself
        # runs under research env (root .venv, where prefect lives), so
        # this subprocess call must cross environments explicitly, not
        # inherit whichever python happens to be on PATH.
        calls = []

        def fake_runner(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return FakeCompletedProcess(returncode=0, stdout="MLFLOW_RUN_ID=abc123\n")

        eval_baseline.run_evaluation_for_variant(
            Path("/repo"), "varon", Path("/staging"), cmd_runner=fake_runner
        )

        [(cmd, kwargs)] = calls
        assert "vendor/starcop/.venv" in cmd[0]
        assert kwargs["cwd"] == Path("/repo")

    def test_always_passes_emit_docs_assets_never_limit(self):
        calls = []

        def fake_runner(cmd, **kwargs):
            calls.append(cmd)
            return FakeCompletedProcess(returncode=0, stdout="MLFLOW_RUN_ID=abc123\n")

        eval_baseline.run_evaluation_for_variant(
            Path("/repo"), "mag1c_only", Path("/staging/run-1"), cmd_runner=fake_runner
        )

        [cmd] = calls
        assert cmd == [
            "/repo/vendor/starcop/.venv/bin/python",
            "/repo/scripts/run_starcop_baseline_evaluation.py",
            "mag1c_only",
            "--emit-docs-assets",
            "/staging/run-1",
        ]
        assert "--limit" not in cmd

    def test_raises_with_variant_and_stderr_tail_on_nonzero_exit(self):
        stderr_lines = [f"line-{index}" for index in range(45)]

        def failing_runner(cmd, **kwargs):
            return FakeCompletedProcess(returncode=1, stderr="\n".join(stderr_lines))

        with pytest.raises(RuntimeError) as error:
            eval_baseline.run_evaluation_for_variant(
                Path("/repo"), "mag1c_rgb", Path("/staging"), cmd_runner=failing_runner
            )

        assert str(error.value) == (
            "evaluation failed for variant='mag1c_rgb' with exit code 1\n"
            + "\n".join(stderr_lines[-40:])
        )

    def test_builds_the_exact_baseline_subprocess_contract(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.delenv("MLFLOW_S3_ENDPOINT_URL", raising=False)
        calls = []

        def fake_runner(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return FakeCompletedProcess(returncode=0, stdout="MLFLOW_RUN_ID=abc123\n")

        eval_baseline.run_evaluation_for_variant(
            Path("/repo"), "varon", Path("/staging"), cmd_runner=fake_runner
        )

        [(cmd, kwargs)] = calls
        assert cmd == [
            "/repo/vendor/starcop/.venv/bin/python",
            "/repo/scripts/run_starcop_baseline_evaluation.py",
            "varon",
            "--emit-docs-assets",
            "/staging",
        ]
        assert kwargs["cwd"] == Path("/repo")
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["env"]["MLFLOW_TRACKING_URI"] == eval_baseline.MLFLOW_TRACKING_URI
        assert kwargs["env"]["MLFLOW_S3_ENDPOINT_URL"] == eval_baseline.MLFLOW_S3_ENDPOINT_URL

    def test_passes_mlflow_tracking_uri_to_the_subprocess_env(self):
        # Real bug, caught live 2026-08-22: run_starcop_baseline_evaluation.py
        # logs to MLflow at the end of its run and needs MLFLOW_TRACKING_URI
        # in its own environment -- the flow process only has this as a
        # Python-level constant (eval_baseline.MLFLOW_TRACKING_URI), which
        # never reaches a subprocess unless explicitly injected into env=.
        # A real 342-scene varon run completed successfully and then crashed
        # at the logging step with exactly this omission.
        calls = []

        def fake_runner(cmd, **kwargs):
            calls.append(kwargs)
            return FakeCompletedProcess(returncode=0, stdout="MLFLOW_RUN_ID=abc123\n")

        eval_baseline.run_evaluation_for_variant(
            Path("/repo"), "varon", Path("/staging"), cmd_runner=fake_runner
        )

        [kwargs] = calls
        assert kwargs["env"]["MLFLOW_TRACKING_URI"] == eval_baseline.MLFLOW_TRACKING_URI

    def test_passes_mlflow_s3_endpoint_url_to_the_subprocess_env(self):
        # Real bug, caught live 2026-08-22: without MLFLOW_S3_ENDPOINT_URL,
        # boto3 defaults to real AWS S3 instead of Backblaze B2, and a B2 key
        # against real AWS fails with InvalidAccessKeyId -- reproduced live
        # and confirmed this was the exact cause. .env.mlflow has always had
        # this var; .env.prefect never did, because the audit that copied
        # AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY over used a grep pattern
        # (`^[A-Z_]*=`) that silently can't match a var name containing a
        # digit, like "S3" in this one -- it never showed up to copy. Fixed
        # as a code constant + explicit injection instead (matching
        # MLFLOW_TRACKING_URI's own precedent), not another .env.prefect
        # line: this is a public endpoint hostname, not a secret.
        calls = []

        def fake_runner(cmd, **kwargs):
            calls.append(kwargs)
            return FakeCompletedProcess(returncode=0, stdout="MLFLOW_RUN_ID=abc123\n")

        eval_baseline.run_evaluation_for_variant(
            Path("/repo"), "varon", Path("/staging"), cmd_runner=fake_runner
        )

        [kwargs] = calls
        assert kwargs["env"]["MLFLOW_S3_ENDPOINT_URL"] == eval_baseline.MLFLOW_S3_ENDPOINT_URL

    def test_subprocess_env_still_inherits_the_rest_of_the_process_environment(self):
        # Injecting MLFLOW_TRACKING_URI must not clobber everything else the
        # subprocess needs (AWS creds, MLFLOW_TRACKING_USERNAME/PASSWORD from
        # .env.prefect, PATH, etc.) -- env= must start from a copy of the
        # real environment, not a bare dict with only the one new key.
        import os

        calls = []

        def fake_runner(cmd, **kwargs):
            calls.append(kwargs)
            return FakeCompletedProcess(returncode=0, stdout="MLFLOW_RUN_ID=abc123\n")

        eval_baseline.run_evaluation_for_variant(
            Path("/repo"), "varon", Path("/staging"), cmd_runner=fake_runner
        )

        [kwargs] = calls
        for key, value in os.environ.items():
            if key not in ("MLFLOW_TRACKING_URI", "MLFLOW_S3_ENDPOINT_URL"):
                assert kwargs["env"].get(key) == value


class TestStartBentomlServe:
    """start_bentoml_serve had zero direct test coverage before this --
    only exercised indirectly via run_live_check_for_variant's tests, which
    fake start_serve_fn entirely and never see its real env= construction.
    That's exactly how the MLFLOW_TRACKING_URI gap above went unnoticed:
    the same bug exists here (service.py's __init__ requires
    MLFLOW_TRACKING_URI in os.environ, os.environ alone doesn't have it)."""

    def test_passes_mlflow_tracking_uri_to_the_subprocess_env(self):
        calls = []

        def fake_popen(cmd, **kwargs):
            calls.append(kwargs)
            return FakeProcess()

        eval_baseline.start_bentoml_serve(Path("/repo"), "mag1c_rgb", 3001, popen=fake_popen)

        [kwargs] = calls
        assert kwargs["env"]["MLFLOW_TRACKING_URI"] == eval_baseline.MLFLOW_TRACKING_URI

    def test_passes_mlflow_s3_endpoint_url_to_the_subprocess_env(self):
        # Same real bug as run_evaluation_for_variant's -- this process also
        # needs to reach B2 (to download the served model's weights via
        # model_loader.py), not just the tracking server.
        calls = []

        def fake_popen(cmd, **kwargs):
            calls.append(kwargs)
            return FakeProcess()

        eval_baseline.start_bentoml_serve(Path("/repo"), "mag1c_rgb", 3001, popen=fake_popen)

        [kwargs] = calls
        assert kwargs["env"]["MLFLOW_S3_ENDPOINT_URL"] == eval_baseline.MLFLOW_S3_ENDPOINT_URL

    def test_pins_model_name_and_stage_for_the_given_variant(self):
        calls = []

        def fake_popen(cmd, **kwargs):
            calls.append(kwargs)
            return FakeProcess()

        eval_baseline.start_bentoml_serve(Path("/repo"), "mag1c_only", 3001, popen=fake_popen)

        [kwargs] = calls
        assert kwargs["env"]["MODEL_NAME"] == "starcop-baseline-mag1c-only"
        assert kwargs["env"]["MODEL_STAGE"] == "Staging"

    def test_uses_the_requested_port(self):
        calls = []

        def fake_popen(cmd, **kwargs):
            calls.append(cmd)
            return FakeProcess()

        eval_baseline.start_bentoml_serve(Path("/repo"), "mag1c_rgb", 3005, popen=fake_popen)

        [cmd] = calls
        assert "src.baselines.starcop.serving.service:MethaneDetectionService" in cmd
        assert "3005" in cmd

    def test_builds_the_exact_bentoml_subprocess_contract(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.delenv("MLFLOW_S3_ENDPOINT_URL", raising=False)
        calls = []

        def fake_popen(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return FakeProcess()

        eval_baseline.start_bentoml_serve(Path("/repo"), "mag1c_rgb", 3005, popen=fake_popen)

        [(cmd, kwargs)] = calls
        assert cmd == [
            "/repo/.venv/bin/python",
            "-m",
            "bentoml",
            "serve",
            "src.baselines.starcop.serving.service:MethaneDetectionService",
            "--port",
            "3005",
        ]
        assert kwargs["cwd"] == Path("/repo")
        assert kwargs["env"]["MODEL_NAME"] == "starcop-baseline-mag1c-rgb"
        assert kwargs["env"]["MODEL_STAGE"] == "Staging"
        assert kwargs["env"]["MLFLOW_TRACKING_URI"] == eval_baseline.MLFLOW_TRACKING_URI
        assert kwargs["env"]["MLFLOW_S3_ENDPOINT_URL"] == eval_baseline.MLFLOW_S3_ENDPOINT_URL


class TestWaitForHealth:
    def test_returns_health_payload_once_it_responds_200(self):
        sleeps = []

        def fake_post(url, timeout=None):
            return FakeResponse(status_code=200, _json_data={"status": "ok", "model_version": "2"})

        result = eval_baseline.wait_for_health(
            "http://localhost:3001",
            http_post=fake_post,
            sleep_fn=sleeps.append,
            time_fn=_counting_clock(),
        )

        assert result == {"status": "ok", "model_version": "2"}

    def test_retries_through_connection_errors_before_succeeding(self):
        import requests

        attempts = {"n": 0}
        calls = []
        sleeps = []

        def fake_post(url, timeout=None):
            calls.append((url, timeout))
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise requests.exceptions.ConnectionError("not up yet")
            return FakeResponse(status_code=200, _json_data={"status": "ok"})

        result = eval_baseline.wait_for_health(
            "http://localhost:3001",
            http_post=fake_post,
            sleep_fn=sleeps.append,
            time_fn=_counting_clock(),
        )

        assert result == {"status": "ok"}
        assert attempts["n"] == 3
        assert calls == [("http://localhost:3001/health", 10)] * 3
        assert sleeps == [2, 2]

    def test_raises_timeout_error_past_the_deadline(self):
        import requests

        def always_failing_post(url, timeout=None):
            raise requests.exceptions.ConnectionError("never up")

        clock_values = iter((0, 0, 100))

        def clock():
            return next(clock_values)

        with pytest.raises(TimeoutError, match="never up"):
            eval_baseline.wait_for_health(
                "http://localhost:3001",
                timeout_seconds=10,
                http_post=always_failing_post,
                sleep_fn=lambda s: None,
                time_fn=clock,
            )

    def test_reports_none_when_the_deadline_expires_before_any_request(self):
        clock_values = iter((0, 0))

        with pytest.raises(TimeoutError, match=r"within 0s \(None\)"):
            eval_baseline.wait_for_health(
                "http://localhost:3001",
                timeout_seconds=0,
                http_post=lambda url, timeout=None: pytest.fail("must not request health"),
                sleep_fn=lambda seconds: None,
                time_fn=lambda: next(clock_values),
            )

    def test_stops_at_the_deadline_without_an_extra_health_request(self):
        import requests

        attempts = []
        clock_values = iter((0, 0, 10, 11))

        def fake_clock():
            return next(clock_values)

        def failing_post(url, timeout=None):
            attempts.append((url, timeout))
            raise requests.exceptions.ConnectionError("not up")

        with pytest.raises(TimeoutError):
            eval_baseline.wait_for_health(
                "http://localhost:3001",
                timeout_seconds=10,
                http_post=failing_post,
                sleep_fn=lambda seconds: None,
                time_fn=fake_clock,
            )

        assert attempts == [("http://localhost:3001/health", 10)]


def _counting_clock(step=1):
    state = {"t": 0}

    def clock():
        state["t"] += step
        return state["t"]

    return clock


class TestRunLiveCheckForVariant:
    def test_ensures_mlflow_s3_endpoint_url_is_set_for_the_in_process_verify_call(
        self, monkeypatch
    ):
        # Real bug, caught live 2026-08-22: unlike run_evaluation_for_variant
        # and start_bentoml_serve (which shell out to a subprocess, so
        # injecting env= reaches them), this function calls verify_fn
        # (live_verify.verify_variant by default) directly in-process --
        # injecting MLFLOW_S3_ENDPOINT_URL into a subprocess env= dict never
        # reaches an in-process call. verify_variant's own MLflow artifact
        # download reads os.environ directly, and without this var boto3
        # defaults to real AWS S3 instead of Backblaze B2 -- reproduced live
        # against a real MLflow run before fixing (a real artifact
        # download succeeded once this var was set, failed without it).
        import os

        monkeypatch.delenv("MLFLOW_S3_ENDPOINT_URL", raising=False)
        process = FakeProcess()

        eval_baseline.run_live_check_for_variant(
            Path("/repo"),
            "mag1c_rgb",
            "https://mlflow.example.com",
            start_serve_fn=lambda repo_root, variant, port: process,
            wait_for_health_fn=lambda base_url: {"status": "ok"},
            verify_fn=lambda variant, **kwargs: {"passed": True, "results": []},
        )

        assert os.environ["MLFLOW_S3_ENDPOINT_URL"] == eval_baseline.MLFLOW_S3_ENDPOINT_URL

    def test_returns_passed_status_when_verify_fn_reports_passed(self):
        process = FakeProcess()

        result = eval_baseline.run_live_check_for_variant(
            Path("/repo"),
            "mag1c_rgb",
            "https://mlflow.example.com",
            start_serve_fn=lambda repo_root, variant, port: process,
            wait_for_health_fn=lambda base_url: {"status": "ok", "model_version": "2"},
            verify_fn=lambda variant, **kwargs: {"passed": True, "results": []},
        )

        assert result["status"] == "passed"
        assert result["detail"] == {"passed": True, "results": []}
        assert process.terminated is True

    def test_forwards_the_exact_live_check_contract(self):
        calls = {}
        process = FakeProcess()

        def start_serve(repo_root, variant, port):
            calls["start"] = (repo_root, variant, port)
            return process

        def wait_for_health(base_url):
            calls["health"] = base_url
            return {"status": "ok"}

        def verify(variant, **kwargs):
            calls["verify"] = (variant, kwargs)
            return {"passed": True, "results": []}

        result = eval_baseline.run_live_check_for_variant(
            Path("/repo"),
            "mag1c_rgb",
            "https://mlflow.example.com",
            start_serve_fn=start_serve,
            wait_for_health_fn=wait_for_health,
            verify_fn=verify,
        )

        assert result["status"] == "passed"
        assert calls["start"] == (Path("/repo"), "mag1c_rgb", eval_baseline.BENTOML_PORT)
        assert calls["health"] == "http://localhost:3001"
        assert calls["verify"] == (
            "mag1c_rgb",
            {
                "base_url": "http://localhost:3001",
                "tracking_uri": "https://mlflow.example.com",
                "test_csv_path": "/repo/data/starcop_raw/test.csv",
                "root_folder": "/repo/data/starcop_raw/STARCOP_test",
            },
        )

    def test_returns_failed_status_when_verify_fn_reports_failure_not_all_scenes_passed(self):
        process = FakeProcess()

        result = eval_baseline.run_live_check_for_variant(
            Path("/repo"),
            "mag1c_rgb",
            "https://mlflow.example.com",
            start_serve_fn=lambda repo_root, variant, port: process,
            wait_for_health_fn=lambda base_url: {"status": "ok"},
            verify_fn=lambda variant, **kwargs: {"passed": False, "results": [{"passed": False}]},
        )

        assert result["status"] == "failed"
        assert process.terminated is True

    def test_returns_not_run_status_without_raising_when_the_server_never_becomes_healthy(self):
        # Non-fatal: a serving-side outage shouldn't block regenerating the
        # numbers (this phase's own design) -- and the process must still
        # be torn down even though the check never got to verify_fn.
        process = FakeProcess()

        def failing_health(base_url):
            raise TimeoutError("never healthy")

        result = eval_baseline.run_live_check_for_variant(
            Path("/repo"),
            "mag1c_rgb",
            "https://mlflow.example.com",
            start_serve_fn=lambda repo_root, variant, port: process,
            wait_for_health_fn=failing_health,
            verify_fn=lambda variant, **kwargs: pytest.fail("verify_fn must not be called"),
        )

        assert result["status"] == "not_run"
        assert process.terminated is True

    def test_returns_not_run_status_when_verify_fn_itself_raises(self):
        process = FakeProcess()

        def failing_verify(variant, **kwargs):
            raise eval_baseline.live_verify.ModelIdentityMismatch("wrong model")

        result = eval_baseline.run_live_check_for_variant(
            Path("/repo"),
            "mag1c_only",
            "https://mlflow.example.com",
            start_serve_fn=lambda repo_root, variant, port: process,
            wait_for_health_fn=lambda base_url: {"status": "ok"},
            verify_fn=failing_verify,
        )

        assert result["status"] == "not_run"
        assert "wrong model" in result["detail"]
        assert process.terminated is True

    def test_terminates_the_process_even_when_start_serve_succeeds_but_nothing_else_does(self):
        process = FakeProcess()

        eval_baseline.run_live_check_for_variant(
            Path("/repo"),
            "mag1c_rgb",
            "https://mlflow.example.com",
            start_serve_fn=lambda repo_root, variant, port: process,
            wait_for_health_fn=lambda base_url: (_ for _ in ()).throw(RuntimeError("boom")),
            verify_fn=lambda variant, **kwargs: pytest.fail("must not reach verify_fn"),
        )

        assert process.terminated is True
        assert process.waited is True

    def test_kills_the_process_when_terminate_does_not_stop_it_in_time(self):
        # "never raises" is documented on run_live_check_for_variant itself
        # -- a hung process.wait(timeout=30) raising subprocess.TimeoutExpired
        # out of the `finally` would break that contract and clobber whatever
        # result/exception was already in flight.
        class HangingFakeProcess(FakeProcess):
            def __init__(self):
                super().__init__()
                self.killed = False
                self.wait_calls = []

            def wait(self, timeout=None):
                self.wait_calls.append(timeout)
                if len(self.wait_calls) == 1:
                    raise subprocess.TimeoutExpired(cmd="bentoml serve", timeout=timeout)
                self.waited = True

            def kill(self):
                self.killed = True

        process = HangingFakeProcess()

        result = eval_baseline.run_live_check_for_variant(
            Path("/repo"),
            "mag1c_rgb",
            "https://mlflow.example.com",
            start_serve_fn=lambda repo_root, variant, port: process,
            wait_for_health_fn=lambda base_url: (_ for _ in ()).throw(RuntimeError("boom")),
            verify_fn=lambda variant, **kwargs: pytest.fail("must not reach verify_fn"),
        )

        assert result["status"] == "not_run"
        assert process.terminated is True
        assert process.killed is True
        assert process.waited is True
        assert process.wait_calls == [30, 30]


class TestValidateRunCompleteness:
    def _complete_results(self):
        return {
            "varon": {"run_id": "run-varon"},
            "mag1c_only": {"run_id": "run-mo", "live_check": {"status": "passed"}},
            "mag1c_rgb": {"run_id": "run-mr", "live_check": {"status": "not_run"}},
        }

    def test_passes_silently_when_everything_is_present(self):
        eval_baseline.validate_run_completeness(self._complete_results())

    def test_raises_when_a_variants_run_id_is_missing(self):
        results = self._complete_results()
        del results["varon"]["run_id"]

        with pytest.raises(ValueError, match="varon"):
            eval_baseline.validate_run_completeness(results)

    def test_raises_when_a_variant_is_missing_entirely(self):
        results = self._complete_results()
        del results["mag1c_only"]

        with pytest.raises(ValueError, match="mag1c_only"):
            eval_baseline.validate_run_completeness(results)

    def test_raises_when_a_servable_variants_live_check_status_is_missing(self):
        results = self._complete_results()
        del results["mag1c_rgb"]["live_check"]

        with pytest.raises(ValueError, match="mag1c_rgb"):
            eval_baseline.validate_run_completeness(results)

    def test_rejects_an_unexpected_servable_variant_without_a_result(self):
        with pytest.raises(ValueError, match="extra"):
            eval_baseline.validate_run_completeness(
                self._complete_results(), servable_variants=("extra",)
            )

    def test_does_not_require_a_live_check_for_varon(self):
        results = self._complete_results()
        # varon never gets a "live_check" key at all -- out of scope per
        # Phase 5, must not be required here either.
        eval_baseline.validate_run_completeness(results)


class TestRenderAggregateComparison:
    def _reference(self):
        return {
            "varon": {
                "citation": "Table 1, page 9",
                "strong_f1score": {"mean": 0.3072, "std": 0.0287},
                "weak_f1score": {"mean": 0.1035, "std": 0.0152},
                "no_plume_FPR": {"mean": 0.8789, "std": 0.0467},
                "auprc": {"mean": 0.1192, "std": 0.0135},
            },
            "mag1c_only": {
                "citation": "Table 2, page 10",
                "strong_f1score": {"mean": 0.7415, "std": 0.061},
                "weak_f1score": {"mean": 0.4757, "std": 0.0417},
                "no_plume_FPR": {"mean": 0.5211, "std": 0.1098},
                "auprc": {"mean": 0.4941, "std": 0.0549},
            },
            "mag1c_rgb": {
                "citation": "Table 2, page 10",
                "strong_f1score": {"mean": 0.8196, "std": 0.0371},
                "weak_f1score": {"mean": 0.4342, "std": 0.0572},
                "no_plume_FPR": {"mean": 0.4366, "std": 0.0736},
                "auprc": {"mean": 0.5199, "std": 0.0276},
            },
        }

    def _metrics(self, **overrides):
        base = {"strong_f1score": 0.5, "weak_f1score": 0.4, "no_plume_FPR": 0.3, "auprc": 0.2}
        base.update(overrides)
        return base

    def test_renders_one_table_for_each_variant(self):
        variant_results = {
            "varon": {"run_id": "r1", "metrics": self._metrics()},
            "mag1c_only": {
                "run_id": "r2",
                "metrics": self._metrics(),
                "live_check": {"status": "passed"},
            },
            "mag1c_rgb": {
                "run_id": "r3",
                "metrics": self._metrics(),
                "live_check": {"status": "failed"},
            },
        }

        rendered = eval_baseline.render_aggregate_comparison(variant_results, self._reference())

        assert "varon" in rendered
        assert "mag1c_only" in rendered
        assert "mag1c_rgb" in rendered
        assert "MultiSTARCOP — Varon ratio" in rendered
        assert "HyperSTARCOP — mag1c only" in rendered
        assert "HyperSTARCOP — mag1c + RGB" in rendered
        assert rendered.count("| Metric | Paper | Reproduced |") == 3

        # Each variant's own status must land directly under its own
        # heading, not just appear somewhere in the rendered text -- a bug
        # that swapped mag1c_only/mag1c_rgb's statuses, or hardcoded one
        # status for every variant, would otherwise slip through.
        lines = rendered.splitlines()

        def _section(heading_substring):
            idx = next(i for i, line in enumerate(lines) if heading_substring in line)
            return lines[idx : idx + 8]

        varon_section = _section("MultiSTARCOP — Varon ratio")
        assert any("out of scope" in line.lower() for line in varon_section)

        mag1c_only_section = _section("HyperSTARCOP — mag1c only")
        assert any("Live API check:** passed" in line for line in mag1c_only_section)

        mag1c_rgb_section = _section("HyperSTARCOP — mag1c + RGB")
        assert any("Live API check:** failed" in line for line in mag1c_rgb_section)

    def test_includes_the_live_check_status_per_servable_variant(self):
        variant_results = {
            "varon": {"run_id": "r1", "metrics": self._metrics()},
            "mag1c_only": {
                "run_id": "r2",
                "metrics": self._metrics(),
                "live_check": {"status": "passed"},
            },
            "mag1c_rgb": {
                "run_id": "r3",
                "metrics": self._metrics(),
                "live_check": {"status": "not_run"},
            },
        }

        rendered = eval_baseline.render_aggregate_comparison(variant_results, self._reference())

        assert "passed" in rendered
        assert "not_run" in rendered

    def test_never_silently_claims_verification_that_did_not_happen(self):
        # varon has no live_check key at all (out of scope) -- the table
        # must say so explicitly, not omit the row or claim "passed".
        variant_results = {
            "varon": {"run_id": "r1", "metrics": self._metrics()},
            "mag1c_only": {
                "run_id": "r2",
                "metrics": self._metrics(),
                "live_check": {"status": "passed"},
            },
            "mag1c_rgb": {
                "run_id": "r3",
                "metrics": self._metrics(),
                "live_check": {"status": "passed"},
            },
        }

        rendered = eval_baseline.render_aggregate_comparison(variant_results, self._reference())

        # Varon's own section must explicitly say it wasn't checked,
        # not omit the status or claim "passed".
        lines = rendered.splitlines()
        varon_header_idx = next(i for i, line in enumerate(lines) if "MultiSTARCOP" in line)
        varon_section = lines[varon_header_idx : varon_header_idx + 8]
        live_check_lines = [line for line in varon_section if "Live API check" in line]
        assert live_check_lines, "expected a live-check status under Varon's heading"
        assert "out of scope" in live_check_lines[0].lower() or "n/a" in live_check_lines[0].lower()

    def test_renders_the_exact_markdown_contract(self):
        variant_results = {
            "varon": {"run_id": "r1", "metrics": self._metrics()},
            "mag1c_only": {
                "run_id": "r2",
                "metrics": self._metrics(),
                "live_check": {"status": "passed"},
            },
            "mag1c_rgb": {
                "run_id": "r3",
                "metrics": self._metrics(),
                "live_check": {"status": "failed"},
            },
        }

        rendered = eval_baseline.render_aggregate_comparison(variant_results, self._reference())

        assert (
            rendered
            == """# Paper comparison — all variants

## MultiSTARCOP — Varon ratio

**Variant:** `varon` · **Live API check:** out of scope (MultiSTARCOP not deployed live)

| Metric | Paper | Reproduced |
| --- | ---: | ---: |
| Strong F1 | 30.72 ± 2.87 | 50.00 |
| Weak F1 | 10.35 ± 1.52 | 40.00 |
| FPR (tile-level) | 87.89 ± 4.67 | 30.00 |
| AUPRC | 11.92 ± 1.35 | 20.00 |

## HyperSTARCOP — mag1c only

**Variant:** `mag1c_only` · **Live API check:** passed

| Metric | Paper | Reproduced |
| --- | ---: | ---: |
| Strong F1 | 74.15 ± 6.10 | 50.00 |
| Weak F1 | 47.57 ± 4.17 | 40.00 |
| FPR (tile-level) | 52.11 ± 10.98 | 30.00 |
| AUPRC | 49.41 ± 5.49 | 20.00 |

## HyperSTARCOP — mag1c + RGB

**Variant:** `mag1c_rgb` · **Live API check:** failed

| Metric | Paper | Reproduced |
| --- | ---: | ---: |
| Strong F1 | 81.96 ± 3.71 | 50.00 |
| Weak F1 | 43.42 ± 5.72 | 40.00 |
| FPR (tile-level) | 43.66 ± 7.36 | 30.00 |
| AUPRC | 51.99 ± 2.76 | 20.00 |
"""
        )

    def test_renders_na_for_missing_reference_metrics_and_variants(self):
        reference = self._reference()
        del reference["varon"]["auprc"]
        reference.pop("mag1c_only")
        variant_results = {
            variant: {"run_id": variant, "metrics": self._metrics()}
            for variant in eval_baseline.VARIANTS
        }

        rendered = eval_baseline.render_aggregate_comparison(variant_results, reference)

        assert rendered.count("| AUPRC | n/a | 20.00 |") == 2
        assert rendered.count("| Strong F1 | n/a | 50.00 |") == 1


class TestPublishStagingDir:
    def test_replaces_canonical_dir_contents_with_stagings(self, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "a.png").write_text("new-a")
        canonical = tmp_path / "canonical"
        canonical.mkdir()
        (canonical / "old.png").write_text("stale")

        eval_baseline.publish_staging_dir(staging, canonical)

        assert (canonical / "a.png").read_text() == "new-a"
        assert not (canonical / "old.png").exists()

    def test_creates_the_canonical_dir_when_it_does_not_exist_yet(self, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "a.png").write_text("new-a")
        canonical = tmp_path / "outer" / "nested" / "canonical"

        eval_baseline.publish_staging_dir(staging, canonical)

        assert (canonical / "a.png").read_text() == "new-a"

    def test_leaves_no_leftover_temp_directories(self, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "a.png").write_text("new-a")
        canonical = tmp_path / "canonical"

        eval_baseline.publish_staging_dir(staging, canonical)

        siblings = {p.name for p in tmp_path.iterdir()}
        assert siblings == {"staging", "canonical"}


class TestBuildSuccessMessage:
    def test_includes_every_variants_run_id(self):
        variant_results = {
            "varon": {"run_id": "run-varon"},
            "mag1c_only": {"run_id": "run-mo", "live_check": {"status": "passed"}},
            "mag1c_rgb": {"run_id": "run-mr", "live_check": {"status": "failed"}},
        }

        message = eval_baseline.build_success_message(variant_results)

        assert message == (
            "eval_baseline completed:\n"
            "varon: run run-varon\n"
            "mag1c_only: run run-mo (live check: passed)\n"
            "mag1c_rgb: run run-mr (live check: failed)"
        )
        assert "run-varon" in message
        assert "run-mo" in message
        assert "run-mr" in message

    def test_includes_live_check_statuses(self):
        variant_results = {
            "varon": {"run_id": "run-varon"},
            "mag1c_only": {"run_id": "run-mo", "live_check": {"status": "passed"}},
            "mag1c_rgb": {"run_id": "run-mr", "live_check": {"status": "failed"}},
        }

        message = eval_baseline.build_success_message(variant_results)

        assert "passed" in message
        assert "failed" in message


class TestRunEvalBaselineCycle:
    def _run(self, **overrides):
        calls = {"notify_message": None}
        fake_client = object()

        def fake_pull(repo_root):
            assert repo_root == Path("/repo")

        def fake_ensure_registered(client):
            assert client is expected_client
            calls["ensured_client"] = client

        def fake_evaluate(repo_root, variant, staging_dir):
            assert repo_root == Path("/repo")
            assert variant in eval_baseline.VARIANTS
            assert staging_dir.is_dir()
            return f"run-{variant}"

        def fake_live_check(repo_root, variant, tracking_uri):
            assert repo_root == Path("/repo")
            assert variant in eval_baseline.SERVABLE_VARIANTS
            assert tracking_uri == "https://mlflow.example.com"
            return {"status": "passed", "detail": {}}

        def fake_aggregate(variant_results, reference):
            assert set(variant_results) == set(eval_baseline.VARIANTS)
            assert all(
                result.get("metrics")
                == {
                    "strong_f1score": 0.5,
                    "weak_f1score": 0.4,
                    "no_plume_FPR": 0.3,
                    "auprc": 0.2,
                }
                for result in variant_results.values()
            )
            assert reference == {"reference": True}
            return "combined markdown"

        def fake_publish(staging_dir, canonical_dir):
            assert staging_dir.is_dir()
            assert {path.name for path in staging_dir.iterdir()} == {"paper_comparison.md"}
            assert (staging_dir / "paper_comparison.md").read_text() == "combined markdown"
            assert canonical_dir == Path("/repo/docs/assets/paper_eval")
            calls["published"] = (staging_dir, canonical_dir)

        def fake_notify(user_key, api_token, message):
            assert user_key == "pu-key"
            assert api_token == "pu-token"
            calls["notify_message"] = message

        expected_client = overrides.pop("expected_client", fake_client)

        def fake_fetch_metrics(client, run_id):
            assert client is expected_client
            assert run_id in {"run-varon", "run-mag1c_only", "run-mag1c_rgb"}
            return {
                "strong_f1score": 0.5,
                "weak_f1score": 0.4,
                "no_plume_FPR": 0.3,
                "auprc": 0.2,
                "ignored": 99,
            }

        kwargs = dict(
            repo_root=Path("/repo"),
            tracking_uri="https://mlflow.example.com",
            reference_metrics_path=Path("/repo/reference.md"),
            canonical_docs_dir=Path("/repo/docs/assets/paper_eval"),
            pushover_user_key="pu-key",
            pushover_api_token="pu-token",
            client=fake_client,
            pull_fn=fake_pull,
            ensure_registered_fn=fake_ensure_registered,
            evaluate_fn=fake_evaluate,
            live_check_fn=fake_live_check,
            aggregate_fn=fake_aggregate,
            publish_fn=fake_publish,
            notify_fn=fake_notify,
            load_reference_fn=lambda path: (
                {"reference": True} if path == Path("/repo/reference.md") else pytest.fail(path)
            ),
            fetch_metrics_fn=fake_fetch_metrics,
        )
        kwargs.update(overrides)
        try:
            result = eval_baseline.run_eval_baseline_cycle(**kwargs)
        finally:
            self._last_calls = calls
        return result, calls

    def test_returns_a_run_id_per_variant(self):
        result, _ = self._run()

        assert result["varon"]["run_id"] == "run-varon"
        assert result["mag1c_only"]["run_id"] == "run-mag1c_only"
        assert result["mag1c_rgb"]["run_id"] == "run-mag1c_rgb"

    def test_runs_the_live_check_only_for_servable_variants(self):
        result, _ = self._run()

        assert "live_check" in result["mag1c_only"]
        assert "live_check" in result["mag1c_rgb"]
        assert "live_check" not in result["varon"]

    def test_publishes_after_a_successful_run(self):
        _, calls = self._run()

        assert calls["published"] == (
            calls["published"][0],  # staging dir path, don't care about the exact tempdir name
            Path("/repo/docs/assets/paper_eval"),
        )

    def test_notifies_on_success(self):
        _, calls = self._run()

        assert "run-varon" in calls["notify_message"]

    def test_notifies_and_reraises_when_pull_fails(self):
        def failing_pull(repo_root):
            raise RuntimeError("dvc pull failed with exit code 1")

        with pytest.raises(RuntimeError, match="dvc pull failed"):
            self._run(pull_fn=failing_pull)

        assert self._last_calls["notify_message"] == (
            "eval_baseline flow failed at step 'pull_dataset': dvc pull failed with exit code 1"
        )

    def test_notifies_and_reraises_when_evaluation_fails(self):
        def failing_evaluate(repo_root, variant, staging_dir):
            raise RuntimeError(f"evaluation failed for variant={variant!r}")

        with pytest.raises(RuntimeError, match="evaluation failed"):
            self._run(evaluate_fn=failing_evaluate)

        assert self._last_calls["notify_message"] == (
            "eval_baseline flow failed at step 'run_evaluation': "
            "evaluation failed for variant='varon'"
        )

    def test_does_not_publish_when_evaluation_fails(self):
        def failing_evaluate(repo_root, variant, staging_dir):
            raise RuntimeError("boom")

        calls = {"published": False}

        def fake_publish(staging_dir, canonical_dir):
            calls["published"] = True

        with pytest.raises(RuntimeError):
            self._run(evaluate_fn=failing_evaluate, publish_fn=fake_publish)

        assert calls["published"] is False

    @pytest.mark.parametrize(
        ("override_name", "expected_step"),
        [
            ("ensure_registered_fn", "ensure_multistarcop_registered"),
            ("live_check_fn", "live_check"),
            ("aggregate_fn", "aggregate"),
            ("publish_fn", "publish"),
        ],
    )
    def test_reports_the_exact_step_when_a_later_step_fails(self, override_name, expected_step):
        def failing_step(*args):
            raise RuntimeError("step exploded")

        with pytest.raises(RuntimeError, match="step exploded"):
            self._run(**{override_name: failing_step})

        assert self._last_calls["notify_message"] == (
            f"eval_baseline flow failed at step '{expected_step}': step exploded"
        )

    def test_creates_the_mlflow_client_when_one_is_not_supplied(self, monkeypatch):
        captured = {}

        def fake_client_factory(tracking_uri):
            captured["tracking_uri"] = tracking_uri
            return captured["client"]

        captured["client"] = object()
        monkeypatch.setattr(eval_baseline, "MlflowClient", fake_client_factory)
        self._run(client=None, expected_client=captured["client"])

        assert captured["tracking_uri"] == "https://mlflow.example.com"

    def test_a_failed_live_check_does_not_fail_the_whole_run(self):
        # Non-fatal by design -- run_live_check_for_variant itself never
        # raises (see its own tests), it returns a "failed"/"not_run"
        # status dict, so the cycle must complete and still publish.
        def failing_status_live_check(repo_root, variant, tracking_uri):
            return {"status": "failed", "detail": {}}

        result, calls = self._run(live_check_fn=failing_status_live_check)

        assert result["mag1c_only"]["live_check"]["status"] == "failed"
        assert "published" in calls

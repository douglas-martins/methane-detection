"""Tests for flows/retrain.py -- the TASK-7.2 retraining flow.

All side-effecting steps (subprocess calls, HTTP calls, MLflow SDK calls)
are injected as fakes here, so this suite never touches a real subprocess,
network, or MLflow server (Test Size: Small). See run_retraining_cycle's
docstring for why the orchestration logic is a plain function, not a
Prefect @flow, for testing purposes.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import retrain


class FakeCompletedProcess(SimpleNamespace):
    returncode: int
    stdout: str = ""
    stderr: str = ""


class FakeResponse(SimpleNamespace):
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def make_outcome(
    stage, run_id="run-123", reasons=None, model_name="starcop-baseline-mag1c-rgb", version="4"
):
    decision = SimpleNamespace(reasons=reasons or [])
    model_version = SimpleNamespace(name=model_name, version=version) if stage else None
    return SimpleNamespace(
        run_id=run_id, stage=stage, decision=decision, model_version=model_version
    )


class TestMlflowTrackingUri:
    def test_matches_train_mac_sh_hardcoded_value(self):
        # scripts/train_mac.sh hardcodes this same URL rather than reading
        # it from an env file (it's a public HTTPS endpoint, not a secret --
        # only the username/password are real secrets). The flow's own
        # process (the mac-mps worker) never sources train_mac.sh's
        # .env.mlflow, so retrain.py must not depend on MLFLOW_TRACKING_URI
        # being present in its own environment either.
        assert retrain.MLFLOW_TRACKING_URI == "https://methane-detection-mlflow.ghostface.tech"


class TestParseRunId:
    def test_extracts_run_id_from_sentinel_line(self):
        stdout = "some log noise\nMLFLOW_RUN_ID=abc123\nmore noise\n"

        assert retrain.parse_run_id(stdout) == "abc123"

    def test_ignores_unrelated_lines_containing_similar_text(self):
        stdout = "MLFLOW_RUN_ID_OLD=wrong\nMLFLOW_RUN_ID=correct\n"

        assert retrain.parse_run_id(stdout) == "correct"

    def test_raises_when_sentinel_missing(self):
        stdout = "training crashed before finishing\n"

        with pytest.raises(RuntimeError, match="MLFLOW_RUN_ID"):
            retrain.parse_run_id(stdout)


class TestPullDataset:
    def test_calls_dvc_pull_in_repo_root(self):
        calls = []

        def fake_runner(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return FakeCompletedProcess(returncode=0)

        retrain.pull_dataset(Path("/repo"), cmd_runner=fake_runner)

        [(cmd, kwargs)] = calls
        assert cmd[-1] == "pull"
        assert "dvc" in cmd[0]
        assert kwargs["cwd"] == Path("/repo")

    def test_raises_on_nonzero_exit(self):
        def failing_runner(cmd, **kwargs):
            return FakeCompletedProcess(returncode=1)

        with pytest.raises(RuntimeError, match="dvc pull failed"):
            retrain.pull_dataset(Path("/repo"), cmd_runner=failing_runner)


class TestRunTraining:
    def test_returns_parsed_run_id_on_success(self):
        def fake_runner(cmd, **kwargs):
            return FakeCompletedProcess(returncode=0, stdout="MLFLOW_RUN_ID=xyz789\n")

        run_id = retrain.run_training(Path("/repo"), cmd_runner=fake_runner)

        assert run_id == "xyz789"

    def test_invokes_train_mac_script_in_repo_root(self):
        calls = []

        def fake_runner(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return FakeCompletedProcess(returncode=0, stdout="MLFLOW_RUN_ID=xyz789\n")

        retrain.run_training(Path("/repo"), cmd_runner=fake_runner)

        [(cmd, kwargs)] = calls
        assert "train_mac.sh" in cmd[0]
        assert kwargs["cwd"] == Path("/repo")

    def test_raises_on_nonzero_exit_without_parsing_stdout(self):
        def failing_runner(cmd, **kwargs):
            return FakeCompletedProcess(returncode=1, stdout="")

        with pytest.raises(RuntimeError, match="training failed"):
            retrain.run_training(Path("/repo"), cmd_runner=failing_runner)

    def test_raises_with_captured_stderr_for_debuggability(self):
        # A real incident (2026-08-18): a bare "exit code 1" with no
        # surfaced output cost real debugging time twice -- had to manually
        # re-run train_mac.sh outside the flow to find the actual error
        # (wandb's no-tty UsageError). The raised message must carry enough
        # of stderr to diagnose the failure straight from Prefect's logs.
        def failing_runner(cmd, **kwargs):
            return FakeCompletedProcess(
                returncode=1,
                stderr="wandb.errors.UsageError: api_key not configured (no-tty)",
            )

        with pytest.raises(RuntimeError, match="api_key not configured"):
            retrain.run_training(Path("/repo"), cmd_runner=failing_runner)


class TestPromote:
    def test_pins_explicit_model_name_not_the_module_default(self, monkeypatch):
        captured = {}

        def fake_decide_and_promote(client, run_id, model_name):
            captured["model_name"] = model_name
            captured["run_id"] = run_id
            return make_outcome(stage="Staging", run_id=run_id)

        monkeypatch.setattr(retrain.promote_model, "decide_and_promote", fake_decide_and_promote)
        monkeypatch.setattr(retrain, "MlflowClient", lambda tracking_uri: object())

        retrain.promote("https://mlflow.example.com", "run-abc")

        assert captured["model_name"] == "starcop-baseline-mag1c-rgb"
        assert captured["run_id"] == "run-abc"
        assert captured["model_name"] != "methane-cnn-starcop"


class TestTriggerCd:
    def test_posts_workflow_dispatch_with_bearer_token(self):
        calls = []

        def fake_post(url, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse(status_code=204)

        retrain.trigger_cd("secret-token", http_post=fake_post)

        [(url, kwargs)] = calls
        assert url == retrain.CD_DISPATCH_URL
        assert kwargs["headers"]["Authorization"] == "Bearer secret-token"
        assert kwargs["json"] == {"ref": "main"}

    def test_raises_on_error_response(self):
        def fake_post(url, **kwargs):
            return FakeResponse(status_code=403)

        with pytest.raises(RuntimeError, match="403"):
            retrain.trigger_cd("bad-token", http_post=fake_post)


class TestNotify:
    def test_posts_message_with_pushover_credentials(self):
        calls = []

        def fake_post(url, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse(status_code=200)

        retrain.notify("user-key", "api-token", "hello", http_post=fake_post)

        [(url, kwargs)] = calls
        assert url == retrain.PUSHOVER_MESSAGES_URL
        assert kwargs["data"]["token"] == "api-token"
        assert kwargs["data"]["user"] == "user-key"
        assert kwargs["data"]["message"] == "hello"


class TestBuildNotificationMessage:
    def test_success_message_includes_stage_and_model_version(self):
        outcome = make_outcome(stage="Production", version="7")

        message = retrain.build_notification_message(outcome)

        assert "Production" in message
        assert "v7" in message
        assert "run-123" in message

    def test_rejection_message_includes_reasons(self):
        outcome = make_outcome(stage=None, reasons=["val_accuracy 0.80 below threshold 0.85"])

        message = retrain.build_notification_message(outcome)

        assert "NOT promoted" in message
        assert "val_accuracy 0.80 below threshold 0.85" in message


class TestRunRetrainingCycle:
    def _run(self, stage, **overrides):
        calls = {"trigger_cd": False, "notify_message": None}

        def fake_pull(repo_root):
            pass

        def fake_train(repo_root):
            return "run-999"

        def fake_promote(tracking_uri, run_id, model_name):
            calls["promote_args"] = (tracking_uri, run_id, model_name)
            return make_outcome(stage=stage, run_id=run_id)

        def fake_trigger_cd(token):
            calls["trigger_cd"] = True
            calls["trigger_cd_token"] = token

        def fake_notify(user_key, api_token, message):
            calls["notify_message"] = message

        kwargs = dict(
            repo_root=Path("/repo"),
            tracking_uri="https://mlflow.example.com",
            github_token="gh-token",
            pushover_user_key="pu-key",
            pushover_api_token="pu-token",
            pull_fn=fake_pull,
            train_fn=fake_train,
            promote_fn=fake_promote,
            trigger_cd_fn=fake_trigger_cd,
            notify_fn=fake_notify,
        )
        kwargs.update(overrides)
        outcome = retrain.run_retraining_cycle(**kwargs)
        return outcome, calls

    def test_passes_pinned_model_name_to_promote_fn(self):
        _, calls = self._run(stage="Staging")

        assert calls["promote_args"] == (
            "https://mlflow.example.com",
            "run-999",
            "starcop-baseline-mag1c-rgb",
        )

    def test_triggers_cd_when_promoted(self):
        _, calls = self._run(stage="Staging")

        assert calls["trigger_cd"] is True
        assert calls["trigger_cd_token"] == "gh-token"

    def test_does_not_trigger_cd_when_not_promoted(self):
        _, calls = self._run(stage=None)

        assert calls["trigger_cd"] is False

    def test_notifies_on_success(self):
        _, calls = self._run(stage="Production")

        assert "Production" in calls["notify_message"]

    def test_notifies_on_rejection(self):
        _, calls = self._run(stage=None)

        assert "NOT promoted" in calls["notify_message"]

    def test_returns_the_promotion_outcome(self):
        outcome, _ = self._run(stage="Staging")

        assert outcome.stage == "Staging"


class TestRunRetrainingCycleFailureHandling:
    """A step raising must still notify (unattended runs have no one
    watching the Prefect UI) and must not swallow the original error."""

    def _run_with_failing_step(self, **overrides):
        notify_calls = []

        def fake_pull(repo_root):
            pass

        def fake_train(repo_root):
            return "run-999"

        def fake_promote(tracking_uri, run_id, model_name):
            return make_outcome(stage="Staging", run_id=run_id)

        def fake_trigger_cd(token):
            pass

        def fake_notify(user_key, api_token, message):
            notify_calls.append(message)

        kwargs = dict(
            repo_root=Path("/repo"),
            tracking_uri="https://mlflow.example.com",
            github_token="gh-token",
            pushover_user_key="pu-key",
            pushover_api_token="pu-token",
            pull_fn=fake_pull,
            train_fn=fake_train,
            promote_fn=fake_promote,
            trigger_cd_fn=fake_trigger_cd,
            notify_fn=fake_notify,
        )
        kwargs.update(overrides)
        return kwargs, notify_calls

    def test_notifies_and_reraises_when_pull_fails(self):
        def failing_pull(repo_root):
            raise RuntimeError("dvc pull failed with exit code 1")

        kwargs, notify_calls = self._run_with_failing_step(pull_fn=failing_pull)

        with pytest.raises(RuntimeError, match="dvc pull failed"):
            retrain.run_retraining_cycle(**kwargs)

        assert len(notify_calls) == 1
        assert "pull_dataset" in notify_calls[0]
        assert "dvc pull failed with exit code 1" in notify_calls[0]

    def test_notifies_and_reraises_when_training_fails(self):
        def failing_train(repo_root):
            raise RuntimeError("training failed with exit code 1")

        kwargs, notify_calls = self._run_with_failing_step(train_fn=failing_train)

        with pytest.raises(RuntimeError, match="training failed"):
            retrain.run_retraining_cycle(**kwargs)

        assert len(notify_calls) == 1
        assert "run_training" in notify_calls[0]

    def test_notifies_and_reraises_when_promote_fails(self):
        def failing_promote(tracking_uri, run_id, model_name):
            raise RuntimeError("mlflow unreachable")

        kwargs, notify_calls = self._run_with_failing_step(promote_fn=failing_promote)

        with pytest.raises(RuntimeError, match="mlflow unreachable"):
            retrain.run_retraining_cycle(**kwargs)

        assert len(notify_calls) == 1
        assert "promote" in notify_calls[0]

    def test_notifies_and_reraises_when_trigger_cd_fails(self):
        def failing_trigger_cd(token):
            raise RuntimeError("HTTP 403")

        kwargs, notify_calls = self._run_with_failing_step(trigger_cd_fn=failing_trigger_cd)

        with pytest.raises(RuntimeError, match="HTTP 403"):
            retrain.run_retraining_cycle(**kwargs)

        assert len(notify_calls) == 1
        assert "trigger_cd" in notify_calls[0]

    def test_does_not_notify_twice_when_a_step_fails(self):
        def failing_train(repo_root):
            raise RuntimeError("boom")

        kwargs, notify_calls = self._run_with_failing_step(train_fn=failing_train)

        with pytest.raises(RuntimeError):
            retrain.run_retraining_cycle(**kwargs)

        # Only the failure notification -- the success/rejection notify
        # call at the end of the happy path must not also fire.
        assert len(notify_calls) == 1

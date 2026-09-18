"""CLI reproducibility tests; HTTP responses below are explicit integration fixtures."""
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import threading
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from vcrectify import cli
from vcrectify.artifacts import load_checkpoint
from vcrectify.data import save_prepared
from vcrectify.demo import synthetic_dataset
from vcrectify.types import Condition, EvidenceItem, Observation, PreparedDataset


def tiny_fixture():
    source = synthetic_dataset()
    genes = source.genes[:2]
    observations = {}
    for index, (key, old) in enumerate(list(source.observations.items())[:6]):
        condition = Condition(key, "SYNTHETIC_CLI_FIXTURE", (genes[index % 2],))
        observations[key] = Observation(condition, old.delta[:2], old.labels[:2], old.qvalues[:2], old.cells[:, :2])
    ids = list(observations)
    return PreparedDataset(genes, source.control_mean[:2], observations,
                           {"initial": ids[:2], "pool": ids[2:4], "test": ids[4:], "validation": []},
                           {"synthetic": True, "purpose": "CLI integration fixture; not measured biology"})


def write_fixture_config(directory, *, summer=False, endpoint=""):
    directory.mkdir(parents=True, exist_ok=True)
    data = tiny_fixture()
    save_prepared(data, directory / "prepared")
    config = {"project_root": ".", "data": {"prepared": "prepared"},
              "numerical": {"name": "reference_ridge", "seed": 17},
              "reasoning": {"name": "reference_memory"}, "knowledge": {"name": "reference"},
              "run": {"budget": 2, "batch_size": 1, "seed": 17, "replay_capacity": 2},
              "output": "run"}
    if summer:
        items = [EvidenceItem("fixture:%s:%s" % (gene, role),
                              "Explicit test fixture text, not a biological annotation.",
                              (gene,), "test://cli-fixture", metadata={"role": role})
                 for gene in data.genes for role in ["perturbation", "response"]]
        (directory / "items.json").write_text(json.dumps([asdict(item) for item in items]), encoding="utf-8")
        (directory / "graph.json").write_text("{}", encoding="utf-8")
        config["knowledge"] = {"items_path": "items.json"}
        config["reasoning"] = {"name": "summer", "backend": "http", "base_url": endpoint,
                               "model": "TEST-ONLY-SUMMER-HTTP-INTEGRATION-FIXTURE",
                               "api_key_env": "VCRECTIFY_TEST_CLI_KEY", "graph_path": "graph.json",
                               "cache_dir": "cache", "response_retries": 0, "seed": 17}
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_blank_endpoint_preflight_never_constructs_models_or_calls_http(tmp_path, capsys):
    config = write_fixture_config(tmp_path, summer=True)
    with patch("vcrectify.llm.urlopen", side_effect=AssertionError("Unexpected network access")) as transport, \
            patch("vcrectify.cli.reasoning_backbone", side_effect=AssertionError("Unexpected model construction")) as factory:
        assert cli.main(["doctor", "--config", str(config)]) == 2
        report = json.loads(capsys.readouterr().out)
        assert not report["ready"]
        assert any("no service URL" in issue for issue in report["issues"])
        assert cli.main(["run", "--config", str(config)]) == 2
        assert "Preflight failed" in capsys.readouterr().err
        transport.assert_not_called()
        factory.assert_not_called()
    assert not (tmp_path / "run").exists()


def test_config_paths_are_project_relative_and_env_key_value_is_not_read(tmp_path, monkeypatch):
    project = tmp_path / "project"
    config_file = write_fixture_config(project, summer=True, endpoint="${VCRECTIFY_TEST_ENDPOINT:}")
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    config["project_root"] = ".."
    settings = project / "settings"
    settings.mkdir()
    config_file = settings / "config.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("VCRECTIFY_TEST_ENDPOINT", "http://127.0.0.1:8765/v1")
    monkeypatch.setenv("VCRECTIFY_TEST_CLI_KEY", "PRIVATE_TEST_SENTINEL_DO_NOT_PERSIST")
    monkeypatch.chdir(tmp_path)
    loaded = cli.load_config(config_file)
    assert loaded["data"]["prepared"] == str((project / "prepared").resolve())
    assert loaded["reasoning"]["graph_path"] == str((project / "graph.json").resolve())
    assert loaded["reasoning"]["cache_dir"] == str((project / "cache").resolve())
    assert loaded["reasoning"]["base_url"] == "http://127.0.0.1:8765/v1"
    assert loaded["reasoning"]["api_key_env"] == "VCRECTIFY_TEST_CLI_KEY"
    assert "PRIVATE_TEST_SENTINEL" not in json.dumps(loaded)


@pytest.mark.parametrize("key", ["api_key", "token", "password", "secret"])
def test_config_rejects_inline_credentials(tmp_path, key):
    config = tmp_path / "bad.yaml"
    config.write_text(yaml.safe_dump({"reasoning": {key: "private-value"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="environment variable"):
        cli.load_config(config)


def test_reference_cli_run_resume_matches_full_run_and_writes_artifacts(tmp_path, capsys):
    config = write_fixture_config(tmp_path)
    assert cli.main(["run", "--config", str(config), "--max-rounds", "1"]) == 0
    partial = json.loads(capsys.readouterr().out)
    assert partial["acquired"] == 1 and not partial["completed"]
    assert cli.main(["run", "--config", str(config), "--resume"]) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["completed"] and resumed["acquired"] == 2
    assert cli.main(["run", "--config", str(config), "--output", str(tmp_path / "full")]) == 0
    full = json.loads(capsys.readouterr().out)
    assert resumed["last_evaluation"] == full["last_evaluation"]
    artifacts = ["run_manifest.json", "events.jsonl", "metrics.json", "heldout_predictions.json",
                 "checkpoint.npz", "summary.json"]
    assert all((tmp_path / "run" / name).is_file() for name in artifacts)
    saved = load_checkpoint(tmp_path / "run" / "checkpoint.npz")
    reference = load_checkpoint(tmp_path / "full" / "checkpoint.npz")
    assert saved["pool"] == reference["pool"]
    assert saved["reliabilities"] == reference["reliabilities"]
    np.testing.assert_array_equal(saved["numerical"]["weights"], reference["numerical"]["weights"])
    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_metadata"]["synthetic"] is True


def test_whole_engine_summer_http_fixture_runs_and_resumes_without_persisting_key(tmp_path, monkeypatch, capsys):
    requests = []
    secret = "PRIVATE_TEST_SENTINEL_DO_NOT_PERSIST"
    monkeypatch.setenv("VCRECTIFY_TEST_CLI_KEY", secret)

    class FixtureHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.headers.get("Authorization") == "Bearer " + secret
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            payload = json.loads(body["messages"][1]["content"])
            requests.append(payload)
            label = (1 if payload["query"]["gene"].endswith("0") else 0) if payload["stage"] == "de" else -1
            generated = {"label": label,
                         "rationale": "Explicit HTTP integration fixture response; no model inference.",
                         "citations": [payload["summaries"][0]["id"]], "case_citations": []}
            response = json.dumps({"choices": [{"message": {"content": json.dumps(generated)}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), FixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = write_fixture_config(tmp_path, summer=True,
                                      endpoint="http://127.0.0.1:%d/v1" % server.server_port)
        assert cli.main(["run", "--config", str(config), "--max-rounds", "1"]) == 0
        assert cli.main(["run", "--config", str(config), "--resume"]) == 0
        captured = capsys.readouterr()
        assert secret not in captured.out + captured.err
        summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
        assert summary["completed"] and summary["acquired"] == 2
        assert {payload["stage"] for payload in requests} == {"de", "direction"}
        assert any(payload["observed_cases"] for payload in requests)
        for path in tmp_path.rglob("*.json"):
            assert secret not in path.read_text(encoding="utf-8")
        assert secret not in (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8")
        checkpoint = load_checkpoint(tmp_path / "run" / "checkpoint.npz")
        assert secret not in repr(checkpoint)
        assert checkpoint["reasoning"]["model_identity"]["model"].startswith("TEST-ONLY")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_release_source_artifacts_and_default_summer_configuration(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    for path in ["pyproject.toml", "README.md", "README.zh-CN.md", "CITATION.cff",
                 "docs/backbones/summer.md", "src/vcrectify/summer_assets.py",
                 "configs/k562_txpert_summer.yaml", "THIRD_PARTY_NOTICES.md"]:
        assert (root / path).is_file(), path
    monkeypatch.delenv("VCRECTIFY_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("VCRECTIFY_LLM_MODEL", raising=False)
    config = cli.load_config(root / "configs/k562_txpert_summer.yaml")
    assert config["reasoning"]["backend"] == "http"
    assert config["reasoning"]["base_url"] == ""
    assert config["reasoning"]["model"] == "Qwen3-8B"
    assert config["reasoning"]["allow_missing_summaries"] is False

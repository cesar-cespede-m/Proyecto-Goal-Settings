"""SDLC-02.1: evidencia fuera de banda mediante variable de GitHub Actions."""

import base64
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.contracts.release_evidence import ReleaseEvidenceUnsignedV1
from core.integration import (
    EvidenceTransportError,
    EvidenceVariablePayload,
    prepare_evidence_variable,
    publish_evidence_variable,
    sign_release_evidence,
)

REPOSITORY = "example/goal_settings-consumer-sandbox"
COMMIT = "a" * 40
NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
ROOT = Path(__file__).parents[2]


def _fixture(tmp_path: Path) -> tuple[Path, Path, bytes]:
    private_key = Ed25519PrivateKey.generate()
    unsigned = ReleaseEvidenceUnsignedV1.model_validate(
        {
            "consumer": {"repository": REPOSITORY, "commit_sha": COMMIT},
            "goal_settings": {"package_version": "0.8.0", "commit_sha": "b" * 40},
            "refs": {
                "project_id": "project",
                "run_id": "run",
                "report_id": "report",
                "context_version_id": "context",
                "context_version_no": 1,
                "strategy_version_id": "strategy",
                "strategy_version_no": 1,
            },
            "hashes": {
                "dataset_sha256": "1" * 64,
                "context_sha256": "2" * 64,
                "strategy_sha256": "3" * 64,
                "model_artifact_sha256": "4" * 64,
                "report_sha256": "5" * 64,
            },
            "approvals": {
                "training": {
                    "approval_id": "training-approval",
                    "stage": "TRAINING",
                    "decision": "APPROVED",
                    "actor_label": "Revisor ML",
                    "approved_at": NOW,
                    "approved_hash": "3" * 64,
                },
                "publication": {
                    "approval_id": "publication-approval",
                    "stage": "PUBLICATION",
                    "decision": "APPROVED",
                    "actor_label": "Responsable producto",
                    "approved_at": NOW,
                    "approved_hash": "5" * 64,
                },
            },
            "result": {
                "selected_model": "LogisticRegression",
                "primary_metric": "f1",
                "holdout_value": 0.8,
                "guardrails_passed": True,
            },
            "validity": {
                "issued_at": NOW,
                "expires_at": NOW + timedelta(days=30),
                "report_status": "PUBLISHED",
                "context_current_at_issuance": True,
                "strategy_current_at_issuance": True,
            },
        }
    )
    evidence = sign_release_evidence(unsigned, private_key)
    raw = (json.dumps(evidence.model_dump(mode="json"), ensure_ascii=False) + "\n").encode()
    evidence_path = tmp_path / "release-evidence.json"
    public_key_path = tmp_path / "public-key.pem"
    evidence_path.write_bytes(raw)
    public_key_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return evidence_path, public_key_path, raw


def test_valid_evidence_is_verified_then_sent_as_base64(tmp_path: Path) -> None:
    evidence_path, public_key_path, raw = _fixture(tmp_path)
    payload = prepare_evidence_variable(
        evidence_path=evidence_path,
        public_key_path=public_key_path,
        expected_repository=REPOSITORY,
        expected_commit=COMMIT,
        now=NOW + timedelta(days=1),
    )
    observed: list[object] = []

    def runner(command, **kwargs):
        observed.extend([command, kwargs])
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    publish_evidence_variable(payload, runner=runner)

    command = observed[0]
    assert command[:3] == ("gh", "secret", "set")
    assert command[4:6] == ("--repo", REPOSITORY)
    assert len(command) == 6  # el valor nunca va como argumento
    kwargs = observed[1]
    assert base64.b64decode(kwargs.pop("input")) == raw
    assert kwargs == {"check": False, "capture_output": True, "text": True}


def test_other_commit_is_rejected_before_github_is_called(tmp_path: Path) -> None:
    evidence_path, public_key_path, _ = _fixture(tmp_path)
    with pytest.raises(EvidenceTransportError) as captured:
        prepare_evidence_variable(
            evidence_path=evidence_path,
            public_key_path=public_key_path,
            expected_repository=REPOSITORY,
            expected_commit="c" * 40,
            now=NOW + timedelta(days=1),
        )
    assert captured.value.code == "EVIDENCE_VERIFY_12"


def test_publish_cli_dry_run_verifies_without_mutating_github(tmp_path: Path) -> None:
    evidence_path, public_key_path, _ = _fixture(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "publish_release_evidence.py"),
            str(evidence_path),
            "--public-key",
            str(public_key_path),
            "--repository",
            REPOSITORY,
            "--commit",
            COMMIT,
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Evidencia preparada fuera del commit" in result.stdout


def test_github_failure_does_not_echo_the_evidence() -> None:
    payload = EvidenceVariablePayload(
        REPOSITORY, COMMIT, "GOAL_SETTINGS_RELEASE_EVIDENCE_B64", "raw"
    )

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="authentication failed")

    with pytest.raises(EvidenceTransportError) as captured:
        publish_evidence_variable(payload, runner=runner)
    assert captured.value.code == "GH_SECRET_UPDATE_FAILED"
    assert "raw" not in str(captured.value)

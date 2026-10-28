"""SDLC-02.2: la copia autónoma del verificador en el sandbox se comporta como el original."""

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.contracts.release_evidence import ReleaseEvidenceUnsignedV1, ReleaseEvidenceV1
from core.integration import sign_release_evidence, verify_release_evidence

ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / "templates" / "goal_settings-consumer-sandbox"
VENDORED = TEMPLATE / "goal_settings" / "verify_release_evidence.py"
REPOSITORY = "example/goal_settings-consumer-sandbox"
COMMIT = "a" * 40
ISSUED_AT = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
SHA = "d" * 64


def _load_vendored():
    spec = importlib.util.spec_from_file_location("sandbox_verifier", VENDORED)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _approval(stage: str, approved_hash: str) -> dict:
    return {
        "approval_id": f"{stage.lower()}-1",
        "stage": stage,
        "decision": "APPROVED",
        "actor_label": "Responsable de prueba",
        "approved_at": "2026-10-20T11:00:00Z",
        "approved_hash": approved_hash,
    }


def _unsigned() -> dict:
    return {
        "schema_version": "release-evidence.v1",
        "consumer": {"repository": REPOSITORY, "commit_sha": COMMIT},
        "goal_settings": {"package_version": "0.8.0", "commit_sha": "b" * 40},
        "refs": {
            "project_id": "p",
            "run_id": "r",
            "report_id": "rep",
            "context_version_id": "c",
            "context_version_no": 1,
            "strategy_version_id": "s",
            "strategy_version_no": 1,
        },
        "hashes": {
            "dataset_sha256": SHA,
            "context_sha256": SHA,
            "strategy_sha256": "1" * 64,
            "model_artifact_sha256": SHA,
            "report_sha256": "2" * 64,
        },
        "approvals": {
            "training": _approval("TRAINING", "1" * 64),
            "publication": _approval("PUBLICATION", "2" * 64),
        },
        "result": {
            "selected_model": "RandomForestClassifier",
            "primary_metric": "f1",
            "holdout_value": 0.8,
            "guardrails_passed": True,
        },
        "validity": {
            "issued_at": ISSUED_AT.isoformat(),
            "expires_at": (ISSUED_AT + timedelta(days=30)).isoformat(),
            "report_status": "PUBLISHED",
            "context_current_at_issuance": True,
            "strategy_current_at_issuance": True,
        },
    }


def _signed(private_key, mutate=None) -> dict:
    value = _unsigned()
    if mutate:
        mutate(value)
    evidence = sign_release_evidence(ReleaseEvidenceUnsignedV1.model_validate(value), private_key)
    return evidence.model_dump(mode="json")


def _tampered(private_key) -> dict:
    value = _signed(private_key)
    value["result"]["holdout_value"] = 0.123
    return value


def _missing_consumer(private_key) -> dict:
    value = _signed(private_key)
    value.pop("consumer")
    return value


def _scenarios():
    return {
        "valid": (lambda key: _signed(key), {}, 0),
        "schema": (_missing_consumer, {}, 10),
        "signature": (_tampered, {}, 11),
        "other_commit": (lambda key: _signed(key), {"commit": "c" * 40}, 12),
        "expired": (
            lambda key: _signed(key),
            {"now": ISSUED_AT + timedelta(days=31)},
            13,
        ),
        "rejected": (
            lambda key: _signed(
                key,
                lambda v: v["approvals"]["publication"].update(decision="REJECTED"),
            ),
            {},
            14,
        ),
        "stale": (
            lambda key: _signed(
                key,
                lambda v: v["validity"].update(strategy_current_at_issuance=False),
            ),
            {},
            15,
        ),
    }


@pytest.mark.parametrize("name", list(_scenarios()))
def test_vendored_verifier_matches_goal_settings_codes(name: str) -> None:
    vendored = _load_vendored()
    private_key = Ed25519PrivateKey.generate()
    build, overrides, expected = _scenarios()[name]
    evidence = build(private_key)
    kwargs = {
        "expected_repository": REPOSITORY,
        "expected_commit": overrides.get("commit", COMMIT),
        "now": overrides.get("now", ISSUED_AT + timedelta(days=1)),
    }

    original, _ = verify_release_evidence(evidence, private_key.public_key(), **kwargs)
    copy, _ = vendored.verify_release_evidence(evidence, private_key.public_key(), **kwargs)

    assert int(original) == expected
    assert int(copy) == expected


def test_vendored_contract_schema_is_identical() -> None:
    vendored = _load_vendored()

    assert vendored.ReleaseEvidenceV1.model_json_schema() == ReleaseEvidenceV1.model_json_schema()


def test_vendored_verifier_is_standalone() -> None:
    source = VENDORED.read_text(encoding="utf-8")

    assert "from core" not in source
    assert "import core" not in source
    assert "streamlit" not in source
    requirements = (TEMPLATE / "goal_settings" / "requirements-verifier.txt").read_text(
        encoding="utf-8"
    )
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for pin in requirements.split():
        assert f'"{pin}"' in pyproject


def test_gate_workflow_never_runs_pull_request_code() -> None:
    workflow = (TEMPLATE / ".github" / "workflows" / "goal_settings-gate.yml").read_text(
        encoding="utf-8"
    )

    assert "pull_request_target:" in workflow
    assert "\n  pull_request:" not in workflow
    assert "ref: ${{ github.event.repository.default_branch }}" in workflow
    # Las variables se imprimen en registros públicos; los secretos se ocultan.
    assert "secrets.GOAL_SETTINGS_RELEASE_EVIDENCE_B64" in workflow
    assert "vars." not in workflow
    assert "needs: goal_settings-gate" in workflow

"""El verificador real de consola funciona sin Streamlit ni acceso a SQLite."""

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.integration import ReleaseEvidenceService
from core.reporting import ReportService
from tests.support.ml_workflow import prepared_strategy

ROOT = Path(__file__).resolve().parents[2]


def test_verifier_cli_accepts_exact_commit_and_rejects_another(tmp_path: Path) -> None:
    connection, repository, service, strategy_id, runtime = prepared_strategy(tmp_path)
    repository.decide_training(
        strategy_version_id=strategy_id,
        decision="APPROVED",
        actor_label="Revisor ML",
    )
    result = service.execute(strategy_id)
    report_id = ReportService(repository).create_template_report(result.run_id)
    repository.decide_publication(
        report_id=report_id,
        decision="APPROVED",
        actor_label="Responsable producto",
    )
    repository.publish_report(report_id)
    private_key = Ed25519PrivateKey.generate()
    evidence = ReleaseEvidenceService(repository, runtime).export(
        report_id=report_id,
        consumer_repository="example/goal_settings-consumer-sandbox",
        consumer_commit="a" * 40,
        goal_settings_version="0.8.0",
        goal_settings_commit="b" * 40,
        actor_label="Responsable release",
        private_key=private_key,
        issued_at=datetime.now(UTC),
    )
    evidence_path = tmp_path / "release-evidence.json"
    public_key_path = tmp_path / "release-public.pem"
    evidence_path.write_text(
        json.dumps(evidence.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8"
    )
    public_key_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    base = [
        sys.executable,
        str(ROOT / "scripts" / "verify_release_evidence.py"),
        str(evidence_path),
        "--public-key",
        str(public_key_path),
        "--expected-repo",
        "example/goal_settings-consumer-sandbox",
        "--expected-commit",
    ]

    accepted = subprocess.run([*base, "a" * 40], cwd=ROOT, capture_output=True, text=True)
    rejected = subprocess.run([*base, "c" * 40], cwd=ROOT, capture_output=True, text=True)

    assert accepted.returncode == 0
    assert "Evidencia válida" in accepted.stdout
    assert rejected.returncode == 12
    assert "otro repositorio o commit" in rejected.stderr
    connection.close()

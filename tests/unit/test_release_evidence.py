"""SDLC-01: contrato firmado, verificador offline y exportación auditable."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.contracts.release_evidence import ReleaseEvidenceUnsignedV1
from core.domain.strategy import StrategyProposalV1
from core.domain.strategy_service import StrategyService
from core.integration import (
    ReleaseEvidenceError,
    ReleaseEvidenceService,
    VerificationCode,
    sign_release_evidence,
    verify_release_evidence,
)
from core.reporting import ReportService
from tests.support.ml_workflow import prepared_strategy

CONSUMER_REPOSITORY = "example/goal_settings-consumer-sandbox"
CONSUMER_COMMIT = "a" * 40
GOAL_SETTINGS_COMMIT = "b" * 40
ISSUED_AT = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


def _published_report(tmp_path: Path):
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
    return connection, repository, runtime, strategy_id, result.run_id, report_id


def _exported(tmp_path: Path):
    connection, repository, runtime, strategy_id, run_id, report_id = _published_report(tmp_path)
    private_key = Ed25519PrivateKey.generate()
    evidence = ReleaseEvidenceService(repository, runtime).export(
        report_id=report_id,
        consumer_repository=CONSUMER_REPOSITORY,
        consumer_commit=CONSUMER_COMMIT,
        goal_settings_version="0.8.0",
        goal_settings_commit=GOAL_SETTINGS_COMMIT,
        actor_label="Responsable release",
        private_key=private_key,
        issued_at=ISSUED_AT,
    )
    return (
        connection,
        repository,
        runtime,
        strategy_id,
        run_id,
        report_id,
        private_key,
        evidence,
    )


def _verify(evidence, private_key, **kwargs):
    return verify_release_evidence(
        evidence.model_dump_json(),
        private_key.public_key(),
        expected_repository=kwargs.get("repository", CONSUMER_REPOSITORY),
        expected_commit=kwargs.get("commit", CONSUMER_COMMIT),
        now=kwargs.get("now", ISSUED_AT + timedelta(days=1)),
    )[0]


def _resign(evidence, private_key, mutate):
    value = evidence.unsigned().model_dump(mode="json")
    mutate(value)
    return sign_release_evidence(ReleaseEvidenceUnsignedV1.model_validate(value), private_key)


def test_exported_evidence_is_minimized_signed_and_audited(tmp_path: Path) -> None:
    (
        connection,
        repository,
        _,
        _,
        run_id,
        report_id,
        private_key,
        evidence,
    ) = _exported(tmp_path)

    assert _verify(evidence, private_key) is VerificationCode.VALID
    assert evidence.refs.run_id == run_id
    assert evidence.refs.report_id == report_id
    assert evidence.validity.expires_at - evidence.validity.issued_at == timedelta(days=30)
    serialized = evidence.model_dump_json()
    for prohibited in (
        "synthetic-fixture.csv",
        "positive",
        "negative",
        '"positive_class"',
        '"target_column"',
    ):
        assert prohibited not in serialized

    audit = connection.execute(
        """
        SELECT detail_json FROM audit_events
        WHERE event_type = 'RELEASE_EVIDENCE_EXPORTED' AND entity_id = ?
        """,
        (report_id,),
    ).fetchone()
    detail = json.loads(audit["detail_json"])
    assert detail["actor_label"] == "Responsable release"
    assert detail["consumer_commit"] == CONSUMER_COMMIT
    assert "signature" not in detail
    connection.close()


def test_verifier_returns_every_contractual_exit_code(tmp_path: Path) -> None:
    values = _exported(tmp_path)
    connection = values[0]
    private_key = values[-2]
    evidence = values[-1]

    broken_schema = evidence.model_dump(mode="json")
    broken_schema.pop("consumer")
    code, _ = verify_release_evidence(
        broken_schema,
        private_key.public_key(),
        expected_repository=CONSUMER_REPOSITORY,
        expected_commit=CONSUMER_COMMIT,
        now=ISSUED_AT,
    )
    assert code is VerificationCode.SCHEMA_INVALID

    tampered = evidence.model_dump(mode="json")
    tampered["result"]["holdout_value"] = 0.123456
    code, _ = verify_release_evidence(
        tampered,
        private_key.public_key(),
        expected_repository=CONSUMER_REPOSITORY,
        expected_commit=CONSUMER_COMMIT,
        now=ISSUED_AT,
    )
    assert code is VerificationCode.SIGNATURE_INVALID

    assert _verify(evidence, private_key, commit="c" * 40) is VerificationCode.CONSUMER_MISMATCH
    assert (
        _verify(evidence, private_key, now=ISSUED_AT + timedelta(days=31))
        is VerificationCode.EXPIRED
    )

    rejected = _resign(
        evidence,
        private_key,
        lambda value: value["approvals"]["publication"].update(decision="REJECTED"),
    )
    assert _verify(rejected, private_key) is VerificationCode.NOT_APPROVED_OR_PUBLISHED

    stale = _resign(
        evidence,
        private_key,
        lambda value: value["validity"].update(strategy_current_at_issuance=False),
    )
    assert _verify(stale, private_key) is VerificationCode.NOT_CURRENT_AT_ISSUANCE
    connection.close()


def test_export_rejects_tampered_artifact(tmp_path: Path) -> None:
    connection, repository, runtime, _, run_id, report_id = _published_report(tmp_path)
    run = repository.get_run(run_id)
    (runtime / Path(run["artifact_path"])).write_bytes(b"not-the-approved-model")

    with pytest.raises(ReleaseEvidenceError) as captured:
        ReleaseEvidenceService(repository, runtime).export(
            report_id=report_id,
            consumer_repository=CONSUMER_REPOSITORY,
            consumer_commit=CONSUMER_COMMIT,
            goal_settings_version="0.8.0",
            goal_settings_commit=GOAL_SETTINGS_COMMIT,
            actor_label="Responsable release",
            private_key=Ed25519PrivateKey.generate(),
            issued_at=ISSUED_AT,
        )
    assert captured.value.code == "ARTIFACT_HASH_MISMATCH"
    connection.close()


def test_export_requires_published_report_and_real_actor(tmp_path: Path) -> None:
    connection, repository, service, strategy_id, runtime = prepared_strategy(tmp_path)
    repository.decide_training(
        strategy_version_id=strategy_id,
        decision="APPROVED",
        actor_label="Revisor ML",
    )
    result = service.execute(strategy_id)
    report_id = ReportService(repository).create_template_report(result.run_id)
    exporter = ReleaseEvidenceService(repository, runtime)
    common = {
        "report_id": report_id,
        "consumer_repository": CONSUMER_REPOSITORY,
        "consumer_commit": CONSUMER_COMMIT,
        "goal_settings_version": "0.8.0",
        "goal_settings_commit": GOAL_SETTINGS_COMMIT,
        "private_key": Ed25519PrivateKey.generate(),
        "issued_at": ISSUED_AT,
    }

    with pytest.raises(ReleaseEvidenceError) as unpublished:
        exporter.export(actor_label="Responsable release", **common)
    assert unpublished.value.code == "REPORT_NOT_PUBLISHED"

    repository.decide_publication(
        report_id=report_id,
        decision="APPROVED",
        actor_label="Responsable producto",
    )
    repository.publish_report(report_id)
    with pytest.raises(ReleaseEvidenceError) as actor:
        exporter.export(actor_label="PENDIENTE", **common)
    assert actor.value.code == "EXPORT_ACTOR_REQUIRED"
    connection.close()


def test_export_rejects_report_after_strategy_is_superseded(tmp_path: Path) -> None:
    connection, repository, runtime, strategy_id, _, report_id = _published_report(tmp_path)
    strategy = repository.get_strategy(strategy_id)
    edited = json.loads(strategy["strategy_json"])
    edited["seed"] += 1
    project_id = repository.get_context(strategy["context_version_id"])["project_id"]
    StrategyService(repository).create_version(
        project_id=project_id,
        proposal=StrategyProposalV1.model_validate(edited),
    )

    with pytest.raises(ReleaseEvidenceError) as captured:
        ReleaseEvidenceService(repository, runtime).export(
            report_id=report_id,
            consumer_repository=CONSUMER_REPOSITORY,
            consumer_commit=CONSUMER_COMMIT,
            goal_settings_version="0.8.0",
            goal_settings_commit=GOAL_SETTINGS_COMMIT,
            actor_label="Responsable release",
            private_key=Ed25519PrivateKey.generate(),
            issued_at=ISSUED_AT,
        )
    assert captured.value.code == "REPORT_NOT_CURRENT"
    connection.close()

"""C2-05: propuestas de cambio sólo como borradores revisables."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.ai_engine.change_proposals import (
    ChangeProposalService,
    EvidenceItem,
)
from core.ai_engine.provider import LLMResponse
from core.contracts.platform import ChangeProposalV1
from core.integration.change_drafts import render_pull_request_draft, write_drafts
from scripts import propose_changes


def _proposal(*, evidence_refs: list[str], title: str = "Revisar el cambio sugerido") -> dict:
    return {
        "schema_version": "change-proposal.v1",
        "kind": "code_change",
        "title": title,
        "rationale": "La evidencia agregada justifica que una persona revise este cambio.",
        "evidence_refs": evidence_refs,
        "delivery": "pull_request",
        "requires_human_review": True,
    }


def _answer(*proposals: dict) -> str:
    return json.dumps({"proposals": list(proposals)}, ensure_ascii=False)


class FakeProvider:
    name, model = "fake", "fake"

    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)
        self.requests = []

    def complete_json(self, request):
        self.requests.append(request)
        return LLMResponse(
            content=self.contents.pop(0),
            provider=self.name,
            model=self.model,
            latency_ms=1,
        )


HEALTH = EvidenceItem(
    ref="health:run-1",
    kind="model_health",
    summary="Salud REVIEW; recomendación PROPOSE_RETRAINING; motivos PERFORMANCE_DROPPED",
)
QUALITY = EvidenceItem(
    ref="quality:file-1.measure",
    kind="data_quality",
    summary="Archivo sample.csv; columna measure; conteos EXTREME_OUTLIER=4",
)


def test_evidence_item_is_strict() -> None:
    with pytest.raises(ValidationError):
        EvidenceItem.model_validate({"ref": "health:1", "kind": "model_health", "summary": 3})
    with pytest.raises(ValidationError):
        EvidenceItem(
            ref="health:1",
            kind="model_health",
            summary="Resumen agregado",
            unexpected="no",
        )


def test_valid_ai_proposal_is_kept_with_strict_schema() -> None:
    provider = FakeProvider(_answer(_proposal(evidence_refs=[HEALTH.ref])))

    result = ChangeProposalService(provider).propose([HEALTH])

    assert result.mode == "LLM"
    assert result.proposals[0].evidence_refs == [HEALTH.ref]
    schema = provider.requests[0].json_schema
    assert schema["additionalProperties"] is False
    assert schema["properties"]["proposals"]["maxItems"] == 3
    assert schema["properties"]["proposals"]["items"]["additionalProperties"] is False


def test_invented_evidence_refs_are_discarded_and_retry_gets_exact_reason() -> None:
    provider = FakeProvider(
        _answer(_proposal(evidence_refs=["health:invented"])),
        _answer(_proposal(evidence_refs=[HEALTH.ref], title="Revisar salud del modelo")),
    )

    result = ChangeProposalService(provider).propose([HEALTH])

    assert result.mode == "LLM"
    assert [item.evidence_refs for item in result.proposals] == [[HEALTH.ref]]
    assert len(provider.requests) == 2
    assert (
        "proposals[0].evidence_refs: «health:invented» no existe en la evidencia entregada"
        in provider.requests[1].user
    )


def test_disabled_mode_uses_deterministic_evidence_backed_fallback() -> None:
    result = ChangeProposalService(None).propose([HEALTH, QUALITY])

    assert result.mode == "DISABLED"
    assert {item.kind for item in result.proposals} == {"code_change", "transform"}
    assert all(item.delivery == "pull_request" for item in result.proposals)
    assert all(item.requires_human_review for item in result.proposals)
    assert {ref for item in result.proposals for ref in item.evidence_refs} == {
        HEALTH.ref,
        QUALITY.ref,
    }


def test_markdown_requires_human_review_and_states_nothing_ran() -> None:
    proposal = ChangeProposalV1.model_validate(_proposal(evidence_refs=[HEALTH.ref]))

    draft = render_pull_request_draft(proposal)

    assert "Checklist de revisión humana" in draft
    assert "- [ ]" in draft
    assert "revisión humana obligatoria" in draft
    assert "No se ejecutó nada" in draft


def test_draft_filenames_are_safe_and_one_per_proposal(tmp_path: Path) -> None:
    proposals = [
        ChangeProposalV1.model_validate(
            _proposal(evidence_refs=[HEALTH.ref], title="../../ Cambio: ágil / seguro?")
        ),
        ChangeProposalV1.model_validate(
            _proposal(evidence_refs=[QUALITY.ref], title="Otro cambio seguro")
        ),
    ]

    paths = write_drafts(proposals, tmp_path / "drafts")

    assert len(paths) == 2
    assert all(path.is_file() and path.parent == tmp_path / "drafts" for path in paths)
    assert all(re.fullmatch(r"\d{2}-[a-z0-9-]+\.md", path.name) for path in paths)
    assert ".." not in paths[0].name and "/" not in paths[0].name


class _Connection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_cli_only_writes_drafts_and_never_invokes_git_or_gh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    connection = _Connection()
    monkeypatch.setattr(
        propose_changes,
        "_open_repository",
        lambda _runtime: (connection, object()),
    )
    monkeypatch.setattr(
        propose_changes, "collect_evidence", lambda _repository, _project, storage_root: [QUALITY]
    )
    monkeypatch.setattr(propose_changes, "build_provider", lambda _settings: None)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("La CLI no debe invocar subprocess, git ni gh")

    monkeypatch.setattr(subprocess, "run", forbidden)
    out = tmp_path / "out"

    assert propose_changes.main(["--project", "project-1", "--out", str(out)]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert connection.closed
    assert len(list(out.glob("*.md"))) == 1
    assert str(next(out.glob("*.md"))) in lines
    assert lines[-1] == "evidence=1 proposals=1 drafts=1"

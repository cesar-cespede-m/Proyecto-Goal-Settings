"""Borradores Markdown para revisión humana de propuestas C2-05."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path

from core.contracts.platform import ChangeProposalV1


def render_pull_request_draft(proposal: ChangeProposalV1) -> str:
    """Renderiza texto revisable; deliberadamente no crea ni publica un pull request."""

    evidence = "\n".join(f"- `{ref}`" for ref in proposal.evidence_refs)
    if not evidence:
        evidence = "- Sin referencias de evidencia adjuntas."
    return (
        f"# {proposal.title}\n\n"
        "## Motivo\n\n"
        f"{proposal.rationale}\n\n"
        "## Evidencia\n\n"
        f"{evidence}\n\n"
        "## Checklist de revisión humana\n\n"
        "- [ ] Una persona verificó que la evidencia respalda el cambio.\n"
        "- [ ] Una persona revisó alcance, seguridad, pruebas y riesgo de datos.\n"
        "- [ ] CI validó la implementación antes de aprobar cualquier publicación.\n\n"
        "> Borrador para revisión humana obligatoria. No se ejecutó nada y no se publicó "
        "ningún cambio.\n"
    )


def _safe_slug(title: str) -> str:
    normalized = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.casefold()).strip("-")
    return slug[:80].rstrip("-") or "propuesta"


def write_drafts(proposals: Sequence[ChangeProposalV1], out_dir: str | Path) -> list[Path]:
    """Escribe un Markdown por propuesta con nombres locales seguros y deterministas."""

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, proposal in enumerate(proposals, start=1):
        path = destination / f"{index:02d}-{_safe_slug(proposal.title)}.md"
        path.write_text(render_pull_request_draft(proposal), encoding="utf-8")
        paths.append(path)
    return paths

"""Servicio del borrador de contexto (CTX-01.6), independiente de Streamlit y del LLM.

El borrador vive asociado a proyecto + archivo + hash. Sólo `finalize` crea una versión de
contexto, y lo hace a través de `BusinessContextService`, que revalida todo contra los datos.
"""

import json
from pathlib import Path

from core.data_engine.models import DatasetProfileV1
from core.db.repositories import WorkflowRepository
from core.domain.assistant import ContextDraftV1
from core.domain.context import BusinessContextV1, ContextVersionResult
from core.domain.context_service import BusinessContextService
from core.domain.errors import ContextValidationError, InvalidStateTransitionError


class StoredDraft:
    """Borrador más la revisión con la que se cargó (para la escritura optimista)."""

    def __init__(self, draft_id: str, draft: ContextDraftV1, stored_revision: int) -> None:
        self.draft_id = draft_id
        self.draft = draft
        self.stored_revision = stored_revision


class ContextDraftService:
    def __init__(self, repository: WorkflowRepository, storage_root: str | Path) -> None:
        self.repository = repository
        self.storage_root = Path(storage_root)

    def _dataset(self, project_id: str, dataset_id: str):
        dataset = self.repository.get_dataset(dataset_id)
        if (
            dataset["project_id"] != project_id
            or dataset["admission_status"] != "ACCEPTED"
            or not dataset["is_active"]
        ):
            raise ContextValidationError(
                "DATASET_NOT_ACTIVE",
                "El borrador requiere el archivo aceptado y activo de este proyecto",
            )
        return dataset

    def open(self, project_id: str, dataset_id: str) -> StoredDraft:
        """Retoma el borrador activo del archivo o crea uno nuevo."""

        dataset = self._dataset(project_id, dataset_id)
        row = self.repository.get_active_draft(project_id, dataset_id, dataset["sha256"])
        if row is not None:
            draft = ContextDraftV1.model_validate_json(row["draft_json"])
            return StoredDraft(row["id"], draft, int(row["revision"]))
        profile = DatasetProfileV1.model_validate(json.loads(dataset["profile_json"]))
        draft = ContextDraftV1.start(
            project_id=project_id,
            dataset_id=dataset_id,
            dataset_sha256=dataset["sha256"],
            profile=profile,
        )
        latest = self.repository.get_latest_context(project_id)
        if latest is not None and latest["dataset_id"] == dataset_id:
            payload = json.loads(latest["context_json"])
            draft.seed_from_context(BusinessContextV1.model_validate(payload["context"]))
        draft_id = self.repository.create_draft(
            project_id=project_id,
            dataset_id=dataset_id,
            dataset_sha256=dataset["sha256"],
            revision=draft.revision,
            draft_json=draft.model_dump_json(),
        )
        return StoredDraft(draft_id, draft, draft.revision)

    def reuse_from_bundle(
        self, stored: StoredDraft, content: bytes, *, filename: str, actor_label: str
    ) -> int:
        """C2-04: trae como propuestas las decisiones de una receta exportada de otro proyecto."""

        from core.domain.recipe_bundle import parse_bundle

        if not actor_label or not actor_label.strip():
            raise ValueError("Indica quién importa la receta")
        bundle, context = parse_bundle(content)
        proposed = stored.draft.reuse_context(context, origin=f"archivo «{filename}»")
        self.save(stored)
        self.repository.record_context_reused(
            project_id=stored.draft.project_id,
            source_context_id=bundle.context_ref.context.id,
            actor_label=actor_label.strip(),
            proposed=proposed,
        )
        return proposed

    def reuse(self, stored: StoredDraft, context_version_id: str, *, actor_label: str) -> int:
        """GA-13: trae como propuestas las decisiones de un contexto de otro proyecto."""

        if not actor_label or not actor_label.strip():
            raise ValueError("Indica quién reutiliza el contexto")
        row = next(
            (
                item
                for item in self.repository.list_reusable_contexts(stored.draft.project_id)
                if item["id"] == context_version_id
            ),
            None,
        )
        if row is None:
            raise LookupError("El contexto guardado no está disponible")
        context = BusinessContextV1.model_validate(json.loads(row["context_json"])["context"])
        proposed = stored.draft.reuse_context(context, origin=str(row["project_name"]))
        self.save(stored)
        self.repository.record_context_reused(
            project_id=stored.draft.project_id,
            source_context_id=context_version_id,
            actor_label=actor_label.strip(),
            proposed=proposed,
        )
        return proposed

    def save(self, stored: StoredDraft) -> None:
        saved = self.repository.update_draft(
            stored.draft_id,
            expected_revision=stored.stored_revision,
            revision=stored.draft.revision,
            draft_json=stored.draft.model_dump_json(),
        )
        if not saved:
            raise InvalidStateTransitionError("DRAFT_CHANGED_ELSEWHERE")
        stored.stored_revision = stored.draft.revision

    def discard(self, stored: StoredDraft) -> None:
        self.repository.close_draft(
            stored.draft_id, status="DISCARDED", project_id=stored.draft.project_id
        )

    def finalize(self, stored: StoredDraft) -> ContextVersionResult:
        """Crea la versión inmutable sólo si todo está confirmado y el archivo no cambió."""

        draft = stored.draft
        dataset = self._dataset(draft.project_id, draft.dataset_id)
        if dataset["sha256"] != draft.dataset_sha256:
            raise ContextValidationError(
                "DRAFT_DATASET_CHANGED",
                "El archivo cambió desde que se inició el borrador",
            )
        context = draft.to_business_context()
        self.save(stored)
        result = BusinessContextService(self.repository, self.storage_root).create_version(
            project_id=draft.project_id,
            dataset_id=draft.dataset_id,
            context=context.model_dump(mode="json"),
        )
        self.repository.close_draft(
            stored.draft_id,
            status="FINALIZED",
            project_id=draft.project_id,
            context_version_id=result.context_version_id,
        )
        return result

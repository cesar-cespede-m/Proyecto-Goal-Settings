"""REPORT-01/AI-01: versiones de reporte desde un run terminado.

La plantilla verificada es siempre la base; el LLM, si está configurado, sólo propone
otra redacción.
"""

import json
from typing import TYPE_CHECKING

from pydantic import ValidationError

from core.db.repositories import WorkflowRepository
from core.domain.canonical import canonical_json
from core.domain.errors import ReportValidationError
from core.domain.models import ReportPayload
from core.domain.states import RunStatus
from core.ml_engine.models import EvaluationSummaryV1, TaskEvaluationV1
from core.reporting.template import build_task_report, build_template_report
from core.reporting.verification import verify_report

if TYPE_CHECKING:
    from core.ai_engine.narrative import NarrativeService

_REPORTABLE = {RunStatus.COMPLETED.value, RunStatus.COMPLETED_WITHOUT_RECOMMENDATION.value}


class ReportService:
    def __init__(
        self,
        repository: WorkflowRepository,
        narrative: "NarrativeService | None" = None,
    ) -> None:
        self.repository = repository
        self.narrative = narrative

    def _report_context(
        self, strategy_version_id: str
    ) -> tuple[str | None, str | None, object | None]:
        strategy_row = self.repository.get_strategy(strategy_version_id)
        strategy = json.loads(strategy_row["strategy_json"])
        dataset = self.repository.get_dataset(str(strategy_row["dataset_id"]))
        classification = json.loads(dataset["profile_json"]).get("data_classification")
        target = strategy.get("target_column")
        return classification, str(target) if target else None, strategy.get("positive_class")

    def build(self, run_id: str) -> ReportPayload:
        run = self.repository.get_run(run_id)
        if run["status"] not in _REPORTABLE:
            raise ReportValidationError(
                "RUN_NOT_REPORTABLE",
                "Sólo un run terminado (con o sin recomendación) puede reportarse",
            )
        stored = json.loads(run["candidate_metrics_json"])
        classification, target_column, positive_class = self._report_context(
            str(run["strategy_version_id"])
        )
        try:
            if stored.get("schema_version") == "task-evaluation.v1":
                payload = build_task_report(
                    run_id=run_id,
                    run_status=str(run["status"]),
                    evaluation=TaskEvaluationV1.model_validate(stored),
                    dataset_classification=classification,
                )
            else:
                payload = build_template_report(
                    run_id=run_id,
                    run_status=str(run["status"]),
                    evaluation=EvaluationSummaryV1.model_validate(stored),
                    dataset_classification=classification,
                    target_column=target_column,
                    positive_class=positive_class,
                )
        except ValidationError as error:
            raise ReportValidationError(
                "EVALUATION_SCHEMA_UNSUPPORTED",
                "El run fue evaluado con un contrato anterior; vuelva a entrenar",
            ) from error
        errors = verify_report(payload)
        if errors:
            raise ReportValidationError(
                "UNVERIFIED_CLAIMS",
                "El reporte contiene cifras sin evidencia: " + ", ".join(errors),
            )
        return payload

    def _store(self, run_id: str, payload: ReportPayload) -> str:
        latest = self.repository.get_latest_report(run_id)
        if latest is not None and latest["payload_json"] == canonical_json(payload):
            return str(latest["id"])
        return self.repository.create_report(run_id=run_id, payload=payload)

    def create_template_report(self, run_id: str) -> str:
        """Crea una versión DRAFT por plantilla; si no cambió, devuelve la vigente."""

        return self._store(run_id, self.build(run_id))

    def create_report(self, run_id: str) -> tuple[str, str, str | None]:
        """Crea el reporte con el modo configurado: (report_id, modo, motivo de respaldo)."""

        template = self.build(run_id)
        if self.narrative is None:
            return self._store(run_id, template), "TEMPLATE_DISABLED", None
        result = self.narrative.narrate(template)
        return self._store(run_id, result.payload), result.mode, result.reason

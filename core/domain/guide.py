"""V2-01: guía determinista del recorrido principal de un proyecto.

La guía sólo interpreta estado persistido. No cambia el proyecto, no llama al LLM y delega
la vigencia de las aprobaciones en los gates de dominio existentes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.data_engine.models import DatasetProfileV1
from core.db.repositories import WorkflowRepository
from core.domain.assistant import ContextDraftV1
from core.domain.gates import has_publication_approval, has_training_approval
from core.domain.states import RunStatus

GuideStepId = Literal["data", "decisions", "plan", "train", "report", "use"]
GuideStepStatus = Literal["done", "current", "locked"]

_STEP_ORDER: tuple[GuideStepId, ...] = (
    "data",
    "decisions",
    "plan",
    "train",
    "report",
    "use",
)
_STEP_TITLES: dict[GuideStepId, str] = {
    "data": "Tus datos",
    "decisions": "Qué quieres anticipar",
    "plan": "Plan y primera aprobación",
    "train": "Entrenar y ver resultados",
    "report": "Reporte y segunda aprobación",
    "use": "Usar el modelo",
}
_LOCKED_REASONS: dict[GuideStepId, str] = {
    "data": "",
    "decisions": "Primero necesitas una tabla aceptada para entrenar.",
    "plan": "Primero guarda las decisiones sobre qué anticipar y qué datos usar.",
    "train": "Primero revisa y aprueba el plan de entrenamiento vigente.",
    "report": "Primero el entrenamiento debe terminar y recomendar un modelo.",
    "use": "Primero una persona debe revisar y aprobar el reporte vigente.",
}
_MANY_MISSING_RATE = 0.30


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class GuideStepV1(_StrictModel):
    id: GuideStepId
    title: str = Field(min_length=1)
    status: GuideStepStatus
    summary: str = Field(min_length=1)
    locked_reason: str | None = None

    @model_validator(mode="after")
    def reason_matches_status(self) -> GuideStepV1:
        if self.status == "locked" and not self.locked_reason:
            raise ValueError("Un paso bloqueado debe explicar por qué")
        if self.status != "locked" and self.locked_reason is not None:
            raise ValueError("Sólo un paso bloqueado puede tener locked_reason")
        return self


class GuideActionV1(_StrictModel):
    label: str = Field(min_length=1)
    step_id: GuideStepId
    reason: str = Field(min_length=1)


class GuideAlertV1(_StrictModel):
    severity: Literal["warning", "info"]
    text: str = Field(min_length=1)
    action_label: str = Field(min_length=1)
    step_id: GuideStepId


class GuideStateV1(_StrictModel):
    project_id: str = Field(min_length=1)
    steps: list[GuideStepV1] = Field(min_length=6, max_length=6)
    current_step: GuideStepId
    next_action: GuideActionV1
    alerts: list[GuideAlertV1] = Field(max_length=3)
    ai_mode: Literal["llm", "rules"]

    @model_validator(mode="after")
    def one_current_step(self) -> GuideStateV1:
        if [step.id for step in self.steps] != list(_STEP_ORDER):
            raise ValueError("La guía debe contener los seis pasos en su orden estable")
        current = [step.id for step in self.steps if step.status == "current"]
        if current != [self.current_step]:
            raise ValueError("La guía debe tener exactamente un paso actual")
        if self.next_action.step_id != self.current_step:
            raise ValueError("La siguiente acción debe llevar al paso actual")
        return self


def _plural_files(count: int) -> str:
    return "1 archivo aceptado" if count == 1 else f"{count} archivos aceptados"


def _summaries(
    *,
    accepted: list,
    active,
    context,
    draft_missing: int | None,
    strategy,
    training_approved: bool,
    run,
    report,
    report_approved: bool,
    use_count: int,
) -> dict[GuideStepId, str]:
    if not accepted:
        data = "No hay archivos aceptados."
    elif active is None:
        data = f"{_plural_files(len(accepted))}; todavía no hay una tabla activa para entrenar."
    else:
        data = (
            f"{_plural_files(len(accepted))}; tabla para entrenar: "
            f"«{active['original_filename_display']}»."
        )

    if context is not None:
        decisions = f"Decisiones guardadas para anticipar «{context['target_column']}»."
    elif draft_missing is not None:
        decisions = f"Borrador en curso; faltan {draft_missing} respuestas por confirmar."
    else:
        decisions = "Todavía no hay decisiones guardadas sobre qué anticipar."

    if strategy is None:
        plan = "Todavía no hay un plan de entrenamiento preparado."
    elif training_approved:
        plan = "El plan de entrenamiento vigente está aprobado."
    else:
        plan = "El plan de entrenamiento está preparado y pendiente de aprobación."

    run_status = None if run is None else str(run["status"])
    train = {
        None: "Todavía no hay un entrenamiento para el plan vigente.",
        RunStatus.RUNNING.value: "El entrenamiento del plan vigente está en curso.",
        RunStatus.COMPLETED.value: "El entrenamiento terminó y recomendó un modelo.",
        RunStatus.COMPLETED_WITHOUT_RECOMMENDATION.value: (
            "El entrenamiento terminó correctamente sin recomendar un modelo."
        ),
        RunStatus.FAILED.value: "El último entrenamiento falló y necesita revisión.",
        RunStatus.BLOCKED.value: "El último entrenamiento quedó bloqueado y necesita revisión.",
    }.get(run_status, "El último entrenamiento tiene un estado que necesita revisión.")

    if report is None:
        report_summary = "Todavía no hay un reporte para el entrenamiento vigente."
    elif report_approved:
        report_summary = "El reporte vigente está aprobado para usar el modelo."
    else:
        report_summary = "El reporte vigente todavía necesita revisión y aprobación."

    if not report_approved:
        use = "El modelo todavía no está aprobado para usarlo."
    elif use_count == 0:
        use = "El modelo está aprobado y todavía no se ha usado con casos nuevos."
    elif use_count == 1:
        use = "El modelo se usó una vez con casos nuevos."
    else:
        use = f"El modelo se usó {use_count} veces con casos nuevos."

    return {
        "data": data,
        "decisions": decisions,
        "plan": plan,
        "train": train,
        "report": report_summary,
        "use": use,
    }


def _linear_steps(current: GuideStepId, summaries: dict[GuideStepId, str]) -> list[GuideStepV1]:
    current_index = _STEP_ORDER.index(current)
    steps: list[GuideStepV1] = []
    for index, step_id in enumerate(_STEP_ORDER):
        if index < current_index:
            status: GuideStepStatus = "done"
            reason = None
        elif index == current_index:
            status = "current"
            reason = None
        else:
            status = "locked"
            reason = _LOCKED_REASONS[step_id]
        steps.append(
            GuideStepV1(
                id=step_id,
                title=_STEP_TITLES[step_id],
                status=status,
                summary=summaries[step_id],
                locked_reason=reason,
            )
        )
    return steps


def _retraining_steps(summaries: dict[GuideStepId, str]) -> list[GuideStepV1]:
    """El ciclo vuelve a entrenar sin borrar los hitos que ya se completaron."""

    return [
        GuideStepV1(
            id=step_id,
            title=_STEP_TITLES[step_id],
            status="current" if step_id == "train" else "done",
            summary=summaries[step_id],
            locked_reason=None,
        )
        for step_id in _STEP_ORDER
    ]


def _column_list(raw: str) -> str:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    return ", ".join(f"«{name}»" for name in names)


def _profile_alerts(active) -> list[GuideAlertV1]:
    if active is None:
        return []
    profile = DatasetProfileV1.model_validate_json(active["profile_json"])
    alerts: list[GuideAlertV1] = []
    described: set[str] = set()

    for raw_warning in profile.warnings:
        warning = raw_warning.removeprefix("DATA_PROFILE:")
        name, _, detail = warning.partition(":")
        columns = _column_list(detail)
        if name == "CONSTANT_COLUMNS" and columns:
            text = (
                f"{columns} tiene el mismo valor en todas las filas y no ayuda a distinguir casos."
            )
        elif name == "POSSIBLE_IDENTIFIER_COLUMNS" and columns:
            text = (
                f"{columns} parece identificar cada caso y podría hacer que el modelo memorice "
                "en vez de aprender."
            )
        elif name == "POSSIBLE_PERSONAL_DATA_COLUMNS" and columns:
            text = (
                f"{columns} parece contener datos personales; conviene usarla sólo si es "
                "necesaria para el objetivo."
            )
        else:
            continue
        if text not in described:
            alerts.append(
                GuideAlertV1(
                    severity="warning",
                    text=text,
                    action_label="Se propondrá excluirla en tus decisiones",
                    step_id="decisions",
                )
            )
            described.add(text)

    many_missing = [
        column.name for column in profile.columns if column.missing_rate >= _MANY_MISSING_RATE
    ]
    if many_missing:
        columns = ", ".join(f"«{name}»" for name in many_missing)
        text = (
            f"{columns} tiene muchos valores vacíos; revisa si aporta información suficiente "
            "antes de usarla."
        )
        if text not in described:
            alerts.append(
                GuideAlertV1(
                    severity="warning",
                    text=text,
                    action_label="Se propondrá excluirla en tus decisiones",
                    step_id="decisions",
                )
            )
    return alerts[:3]


def _health_proposes_retraining(repository: WorkflowRepository, run) -> bool:
    if run is None or not repository.list_scorings(str(run["id"])):
        return False
    # ``health`` sólo lee el run y sus usos; el storage no participa en este cálculo.
    from pathlib import Path

    from core.ml_engine.scoring import ScoringService

    health = ScoringService(repository, Path(".")).health(str(run["id"]))
    return health.recommendation == "PROPOSE_RETRAINING"


def project_guide(
    repository: WorkflowRepository, project_id: str, *, ai_enabled: bool
) -> GuideStateV1:
    """Devuelve dónde está el proyecto, qué sigue y por qué, sin modificarlo."""

    accepted = repository.list_datasets(project_id, admission_status="ACCEPTED")
    active = repository.get_active_dataset(project_id)
    context = repository.get_latest_context(project_id) if active is not None else None
    draft_missing: int | None = None
    if active is not None and context is None:
        draft = repository.get_active_draft(project_id, active["id"], active["sha256"])
        if draft is not None:
            draft_missing = len(
                ContextDraftV1.model_validate_json(draft["draft_json"]).missing_confirmations()
            )

    strategy = repository.get_latest_strategy(context["id"]) if context is not None else None
    training_approved = strategy is not None and has_training_approval(
        repository.connection, str(strategy["id"])
    )
    run = repository.get_latest_run(strategy["id"]) if strategy is not None else None
    report = repository.get_latest_report(run["id"]) if run is not None else None
    report_approved = report is not None and has_publication_approval(
        repository.connection, str(report["id"])
    )
    usable_run = (
        run is not None
        and str(run["status"]) == RunStatus.COMPLETED.value
        and bool(run["artifact_path"])
        and report_approved
    )
    scorings = repository.list_scorings(str(run["id"])) if usable_run else []

    summaries = _summaries(
        accepted=accepted,
        active=active,
        context=context,
        draft_missing=draft_missing,
        strategy=strategy,
        training_approved=training_approved,
        run=run,
        report=report,
        report_approved=report_approved,
        use_count=len(scorings),
    )

    run_status = None if run is None else str(run["status"])
    retraining = False
    if not accepted:
        current: GuideStepId = "data"
        action = GuideActionV1(
            label="Subir tus datos",
            step_id=current,
            reason="Necesitamos al menos un archivo aceptado antes de definir qué anticipar.",
        )
    elif context is None:
        current = "decisions"
        if draft_missing is None:
            reason = (
                "Definir el resultado y el momento de la predicción permite preparar un plan "
                "que responda a tu objetivo."
            )
        else:
            reason = (
                f"Faltan {draft_missing} respuestas por confirmar antes de preparar un plan "
                "que respete tus decisiones."
            )
        action = GuideActionV1(
            label="Definir qué quieres anticipar", step_id=current, reason=reason
        )
    elif strategy is None:
        current = "plan"
        action = GuideActionV1(
            label="Preparar el plan de entrenamiento",
            step_id=current,
            reason=(
                "Tus decisiones ya están guardadas y ahora deben convertirse en un plan "
                "verificable."
            ),
        )
    elif not training_approved:
        current = "plan"
        action = GuideActionV1(
            label="Revisar y aprobar el plan",
            step_id=current,
            reason=(
                "Una persona debe confirmar el plan vigente antes de que el entrenamiento "
                "pueda empezar."
            ),
        )
    elif run is None:
        current = "train"
        action = GuideActionV1(
            label="Entrenar",
            step_id=current,
            reason="El plan vigente ya está aprobado y tiene permiso para entrenar.",
        )
    elif run_status == RunStatus.RUNNING.value:
        current = "train"
        action = GuideActionV1(
            label="Revisar el entrenamiento en curso",
            step_id=current,
            reason="El entrenamiento ya comenzó; revisa su estado antes de iniciar otra acción.",
        )
    elif run_status == RunStatus.FAILED.value:
        current = "train"
        action = GuideActionV1(
            label="Revisar por qué falló el entrenamiento",
            step_id=current,
            reason=(
                "Revisar los datos y el plan permite corregir el problema antes de volver a "
                "entrenar."
            ),
        )
    elif run_status == RunStatus.BLOCKED.value:
        current = "train"
        action = GuideActionV1(
            label="Revisar qué bloquea el entrenamiento",
            step_id=current,
            reason="El entrenamiento no puede continuar hasta resolver el bloqueo indicado.",
        )
    elif run_status == RunStatus.COMPLETED_WITHOUT_RECOMMENDATION.value:
        current = "train"
        action = GuideActionV1(
            label="Revisar qué datos sumar",
            step_id=current,
            reason=(
                "Terminar sin recomendar un modelo es un resultado válido; revisar más datos o "
                "mejores variables es el siguiente paso."
            ),
        )
    elif not report_approved:
        current = "report"
        action = GuideActionV1(
            label="Revisar y aprobar el reporte",
            step_id=current,
            reason=(
                "El entrenamiento recomendó un modelo, pero una persona debe revisar sus "
                "resultados antes de usarlo."
            ),
        )
    elif _retrained_without_comparison(repository, project_id, run):
        current = "use"
        action = GuideActionV1(
            label="Comparar con el modelo anterior",
            step_id=current,
            reason=(
                "Este proyecto reentrena un modelo que sigue en uso. Antes de reemplazarlo, "
                "compara ambos con los mismos casos recientes en «Usar el modelo»."
            ),
        )
    elif not scorings:
        current = "use"
        action = GuideActionV1(
            label="Usar el modelo con casos nuevos",
            step_id=current,
            reason=(
                "El modelo y su reporte ya están aprobados; probar casos nuevos permite "
                "convertirlo en una decisión."
            ),
        )
    elif _health_proposes_retraining(repository, run):
        current = "train"
        retraining = True
        action = GuideActionV1(
            label="Reentrenar",
            step_id=current,
            reason=(
                "Los usos registrados indican que conviene preparar una versión con datos más "
                "recientes."
            ),
        )
    else:
        current = "use"
        action = GuideActionV1(
            label="Decidir con el modelo y medir si ayudó",
            step_id=current,
            reason=(
                "Ya hay usos registrados; anotar sus resultados permite saber si el modelo ayudó."
            ),
        )

    steps = _retraining_steps(summaries) if retraining else _linear_steps(current, summaries)
    return GuideStateV1(
        project_id=project_id,
        steps=steps,
        current_step=current,
        next_action=action,
        # Los avisos anuncian lo que se propondrá al decidir; con decisiones guardadas ya no
        # aplican (las columnas quedaron dentro o fuera por decisión de la persona).
        alerts=_profile_alerts(active) if context is None else [],
        ai_mode="llm" if ai_enabled else "rules",
    )


def _retrained_without_comparison(repository: WorkflowRepository, project_id: str, run) -> bool:
    """V2-26: proyecto de reentrenamiento cuyo modelo nuevo aún no se comparó con el anterior."""

    if run is None:
        return False
    is_retraining = any(
        link["project_id"] == project_id for link in repository.list_retraining_links()
    )
    if not is_retraining:
        return False
    compared = repository.connection.execute(
        "SELECT 1 FROM audit_events WHERE event_type = 'MODEL_COMPARISON' AND entity_id = ? "
        "LIMIT 1",
        (str(run["id"]),),
    ).fetchone()
    return compared is None

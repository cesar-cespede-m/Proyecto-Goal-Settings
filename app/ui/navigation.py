"""Modelo presentacional puro del progreso del journey P0."""

from dataclasses import dataclass
from enum import StrEnum


class StepStatus(StrEnum):
    COMPLETE = "complete"
    CURRENT = "current"
    LOCKED = "locked"


@dataclass(frozen=True)
class WorkflowProgress:
    has_project: bool = False
    has_dataset: bool = False
    has_context: bool = False
    has_training_approval: bool = False
    has_reportable_run: bool = False
    report_approved: bool = False
    report_published: bool = False
    model_usable: bool = False
    has_scoring: bool = False


@dataclass(frozen=True)
class NavigationItem:
    number: int
    title: str
    anchor: str
    status: StepStatus
    requirement: str


def build_navigation(progress: WorkflowProgress) -> tuple[NavigationItem, ...]:
    """Agrupa la preparación P0 y el uso continuo autorizado del modelo."""

    completions = (
        progress.has_dataset,
        progress.has_training_approval,
        progress.has_reportable_run,
        progress.report_approved or progress.report_published,
        progress.has_scoring,
    )
    prerequisites = (
        True,
        progress.has_dataset,
        progress.has_training_approval,
        progress.has_reportable_run,
        progress.model_usable,
    )
    definitions = (
        ("Proyecto y archivo", "proyecto-y-datos", "Crea un proyecto y sube su archivo."),
        (
            "Qué quieres anticipar",
            "contexto-y-estrategia",
            "Primero sube el archivo del proyecto.",
        ),
        (
            "Entrenar y ver resultados",
            "entrenamiento-y-resultados",
            "Primero aprueba cómo se entrenará.",
        ),
        (
            "Reporte y aprobación final",
            "insights-y-cierre",
            "Primero termina un entrenamiento.",
        ),
        (
            "Usar con casos nuevos",
            "uso-continuo",
            "Primero aprueba el reporte.",
        ),
    )
    items: list[NavigationItem] = []
    current_assigned = False
    for index, ((title, anchor, requirement), complete, available) in enumerate(
        zip(definitions, completions, prerequisites, strict=True),
        start=1,
    ):
        if complete:
            status = StepStatus.COMPLETE
        elif available and not current_assigned:
            status = StepStatus.CURRENT
            current_assigned = True
        else:
            status = StepStatus.LOCKED
        items.append(NavigationItem(index, title, anchor, status, requirement))
    return tuple(items)

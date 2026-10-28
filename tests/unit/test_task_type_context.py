"""GA-07d-1 (ADR-016): decisiones por tipo de modelo; tipos sin flujo completo se rechazan."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from core.db.repositories import WorkflowRepository
from core.domain.context import BusinessContextV1
from core.domain.errors import StrategyValidationError
from core.domain.strategy import CANDIDATES_BY_TASK
from core.domain.strategy_service import StrategyService
from tests.unit.test_context import _context_service_for_csv, _simple_context

LEVELS = ["bajo", "medio", "alto"]


def _context(task_type: str, metric: str, positive_class: object = None) -> dict:
    context = _simple_context()
    context.update(task_type=task_type, primary_metric=metric, positive_class=positive_class)
    return context


def test_binary_contexts_are_unchanged() -> None:
    model = BusinessContextV1.model_validate(_simple_context())
    assert model.task_type == "binary_classification" and model.positive_class == 1


@pytest.mark.parametrize(
    ("task_type", "metric", "positive_class"),
    [
        ("binary_classification", "recall", None),  # falta el evento
        ("multiclass_classification", "macro_f1", "alto"),  # sólo sí/no tiene evento
        ("multiclass_classification", "mae", None),  # métrica de otro tipo
        ("regression", "recall", None),
    ],
)
def test_fields_must_match_the_task_type(task_type, metric, positive_class) -> None:
    with pytest.raises(ValidationError):
        BusinessContextV1.model_validate(_context(task_type, metric, positive_class))


def _rows(target) -> bytes:
    return (
        "signal,target\n" + "\n".join(f"{i % 7},{target(i)}" for i in range(60)) + "\n"
    ).encode()


@pytest.mark.parametrize(
    ("task_type", "metric", "target", "classes"),
    [
        ("multiclass_classification", "macro_f1", lambda i: LEVELS[i % 3], 3),
        ("regression", "mae", lambda i: i * 1.5, 0),
    ],
)
def test_new_types_are_validated_and_get_a_strategy_of_their_type(
    tmp_path: Path, task_type, metric, target, classes
) -> None:
    connection, service, project_id, dataset_id = _context_service_for_csv(tmp_path, _rows(target))
    try:
        result = service.create_version(
            project_id=project_id, dataset_id=dataset_id, context=_context(task_type, metric)
        )
        assert len(result.validation.target_distribution) == classes
        template = StrategyService(WorkflowRepository(connection)).build_template(
            project_id=project_id, context_version_id=result.context_version_id
        )
        assert template.task_type == task_type
        assert template.candidates == CANDIDATES_BY_TASK[task_type]
    finally:
        connection.close()


def test_types_outside_the_flow_are_rejected_before_any_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "core.domain.strategy_service.FLOW_READY_TASK_KINDS", {"binary_classification"}
    )
    connection, service, project_id, dataset_id = _context_service_for_csv(
        tmp_path, _rows(lambda i: i * 1.5)
    )
    try:
        result = service.create_version(
            project_id=project_id, dataset_id=dataset_id, context=_context("regression", "mae")
        )
        with pytest.raises(StrategyValidationError) as blocked:
            StrategyService(WorkflowRepository(connection)).build_template(
                project_id=project_id, context_version_id=result.context_version_id
            )
        assert blocked.value.code == "UNSUPPORTED_TASK_KIND"
    finally:
        connection.close()

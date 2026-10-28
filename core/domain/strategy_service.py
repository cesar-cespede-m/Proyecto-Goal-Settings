"""Propuesta determinista STRAT-01, sin LLM ni ejecución de modelos."""

import json
from collections.abc import Callable
from typing import Any

from core.data_engine.models import DatasetProfileV1
from core.db.repositories import WorkflowRepository
from core.domain.canonical import canonical_json
from core.domain.context import BusinessContextV1, ContextValidationSnapshot
from core.domain.errors import StrategyValidationError
from core.domain.strategy import (
    CANDIDATES_BY_TASK,
    FLOW_READY_TASK_KINDS,
    StrategyProposalV1,
    StrategyVersionResult,
)


class StrategyService:
    def __init__(self, repository: WorkflowRepository) -> None:
        self.repository = repository

    def _parents(
        self,
        *,
        project_id: str,
        context_version_id: str,
    ) -> tuple[Any, Any, BusinessContextV1, ContextValidationSnapshot, DatasetProfileV1]:
        context_row = self.repository.get_context(context_version_id)
        if context_row["project_id"] != project_id:
            raise StrategyValidationError(
                "PROJECT_CONTEXT_MISMATCH",
                "El contexto no pertenece al proyecto indicado",
            )
        latest = self.repository.get_latest_context(project_id)
        if latest is None or latest["id"] != context_version_id:
            raise StrategyValidationError(
                "CONTEXT_NOT_CURRENT",
                "Solo el contexto vigente puede originar una estrategia",
            )
        dataset = self.repository.get_dataset(context_row["dataset_id"])
        if dataset["admission_status"] != "ACCEPTED":
            raise StrategyValidationError(
                "DATASET_NOT_ACCEPTED",
                "La estrategia requiere un dataset aceptado",
            )
        payload = json.loads(context_row["context_json"])
        context = BusinessContextV1.model_validate(payload["context"])
        validation = ContextValidationSnapshot.model_validate(payload["validation"])
        profile = DatasetProfileV1.model_validate(json.loads(dataset["profile_json"]))
        return context_row, dataset, context, validation, profile

    @staticmethod
    def _normalize(proposal: StrategyProposalV1) -> StrategyProposalV1:
        value = proposal.model_dump(mode="json")
        value["feature_columns"] = sorted(value["feature_columns"], key=str.casefold)
        value["excluded_columns"] = sorted(value["excluded_columns"], key=str.casefold)
        value["candidates"] = list(CANDIDATES_BY_TASK[proposal.task_type])
        value["warnings"] = sorted(set(value["warnings"]))
        return StrategyProposalV1.model_validate(value)

    @staticmethod
    def _validate_split_support(
        proposal: StrategyProposalV1,
        validation: ContextValidationSnapshot,
    ) -> None:
        fractions = (
            proposal.split.train,
            proposal.split.validation,
            proposal.split.test,
        )
        for class_support in validation.target_distribution:
            if any(int(class_support.count * fraction) < 1 for fraction in fractions):
                raise StrategyValidationError(
                    "INSUFFICIENT_CLASS_SUPPORT",
                    "Cada clase debe aportar ejemplos a train, validation y test",
                )

    def build_template(
        self,
        *,
        project_id: str,
        context_version_id: str,
    ) -> StrategyProposalV1:
        context_row, dataset, context, validation, profile = self._parents(
            project_id=project_id,
            context_version_id=context_version_id,
        )
        if context.task_type not in FLOW_READY_TASK_KINDS:
            # ADR-016: un tipo entra al flujo sólo cuando tiene evaluación, reporte, uso y
            # monitoreo propios; se rechaza antes de cualquier aprobación.
            raise StrategyValidationError(
                "UNSUPPORTED_TASK_KIND",
                "Este tipo de modelo todavía no se puede entrenar desde el flujo con aprobaciones",
            )
        profile_warnings = [f"DATA_PROFILE:{warning}" for warning in profile.warnings]
        proposal = StrategyProposalV1(
            dataset_id=dataset["id"],
            dataset_sha256=dataset["sha256"],
            context_version_id=context_version_id,
            context_sha256=context_row["context_sha256"],
            task_type=context.task_type,
            target_column=context.target_column,
            positive_class=context.positive_class,
            feature_columns=context.allowed_features,
            excluded_columns=context.excluded_features,
            primary_metric=context.primary_metric,
            guardrails=context.guardrails,
            split={"train": 0.6, "validation": 0.2, "test": 0.2},
            preprocessing={},
            candidates=CANDIDATES_BY_TASK[context.task_type],
            model_params={},
            seed=42,
            caps={},
            warnings=["DETERMINISTIC_TEMPLATE_NO_LLM", *profile_warnings],
        )
        normalized = self._normalize(proposal)
        self._validate_split_support(normalized, validation)
        return normalized

    def create_version(
        self,
        *,
        project_id: str,
        proposal: StrategyProposalV1 | dict[str, Any],
        operation_id: str | None = None,
        actor_label: str | None = None,
        verify_current: Callable[[], None] | None = None,
    ) -> StrategyVersionResult:
        model = self._normalize(StrategyProposalV1.model_validate(proposal))
        context_row, dataset, context, validation, _ = self._parents(
            project_id=project_id,
            context_version_id=model.context_version_id,
        )
        expected = {
            "dataset_id": dataset["id"],
            "dataset_sha256": dataset["sha256"],
            "context_sha256": context_row["context_sha256"],
            "target_column": context.target_column,
            "positive_class": canonical_json(context.positive_class),
            "feature_columns": sorted(context.allowed_features, key=str.casefold),
            "excluded_columns": sorted(context.excluded_features, key=str.casefold),
            "primary_metric": context.primary_metric.value,
            "guardrails": canonical_json(context.guardrails),
        }
        actual = {
            "dataset_id": model.dataset_id,
            "dataset_sha256": model.dataset_sha256,
            "context_sha256": model.context_sha256,
            "target_column": model.target_column,
            "positive_class": canonical_json(model.positive_class),
            "feature_columns": model.feature_columns,
            "excluded_columns": model.excluded_columns,
            "primary_metric": model.primary_metric.value,
            "guardrails": canonical_json(model.guardrails),
        }
        mismatches = sorted(key for key in expected if expected[key] != actual[key])
        if mismatches:
            raise StrategyValidationError(
                "STRATEGY_CONTEXT_MISMATCH",
                "La estrategia contradice el contexto en: " + ", ".join(mismatches),
            )
        self._validate_split_support(model, validation)
        strategy_json = canonical_json(model)
        latest = self.repository.get_latest_strategy(model.context_version_id)
        if operation_id is None and latest is not None and latest["strategy_json"] == strategy_json:
            raise StrategyValidationError(
                "STRATEGY_UNCHANGED",
                "La estrategia es idéntica a la versión vigente",
            )
        strategy_id = self.repository.add_strategy(
            context_version_id=model.context_version_id,
            dataset_id=model.dataset_id,
            version_no=None,
            payload=model.model_dump(mode="json"),
            schema_version=model.schema_version,
            source=model.source,
            operation_id=operation_id,
            actor_label=actor_label,
            verify_current=verify_current,
        )
        stored = self.repository.get_strategy(strategy_id)
        return StrategyVersionResult(
            strategy_version_id=strategy_id,
            context_version_id=model.context_version_id,
            dataset_id=model.dataset_id,
            version_no=int(stored["version_no"]),
            fingerprint=stored["fingerprint"],
            proposal=model,
        )

    def create_template_version(
        self,
        *,
        project_id: str,
        context_version_id: str,
    ) -> StrategyVersionResult:
        proposal = self.build_template(
            project_id=project_id,
            context_version_id=context_version_id,
        )
        return self.create_version(project_id=project_id, proposal=proposal)

"""Contratos del asistente de contexto (CTX-01.6), independientes de Streamlit y del LLM.

- `SafeProfileForLLMV1`: lo único que puede salir del equipo hacia un proveedor LLM.
- `AssistantTurnV1`: lo único que el LLM puede devolver; propone, nunca confirma.
- `ContextDraftV1`: borrador con valor, origen y estado por campo. Sólo un borrador con
  todos los campos confirmados por la persona se convierte en `BusinessContextV1`.
"""

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from core.data_engine.models import DatasetProfileV1
from core.domain.context import (
    BusinessContextV1,
    FeatureUse,
    GuardrailMode,
    StrictModel,
)


class DraftField(StrEnum):
    OBJECTIVE = "objective"
    TARGET_COLUMN = "target_column"
    POSITIVE_CLASS = "positive_class"
    PREDICTION_MOMENT = "prediction_moment"
    FALSE_POSITIVE_CONSEQUENCE = "false_positive_consequence"
    FALSE_NEGATIVE_CONSEQUENCE = "false_negative_consequence"
    PRIMARY_METRIC = "primary_metric"
    METRIC_RATIONALE = "metric_rationale"
    DECISION_ACTOR_LABEL = "decision_actor_label"
    GUARDRAIL_MODE = "guardrail_mode"
    GUARDRAIL_METRIC = "guardrail_metric"
    GUARDRAIL_THRESHOLD = "guardrail_threshold"
    GUARDRAIL_RATIONALE = "guardrail_rationale"
    # ADR-016: sí/no, varias clases o cantidades. Vacío equivale a sí/no (borradores previos).
    TASK_TYPE = "task_type"


# Campos que el LLM nunca propone: el evento a detectar es un valor de celda (no sale del
# equipo), el responsable es una declaración personal y el umbral es una decisión humana.
HUMAN_ONLY_FIELDS = frozenset(
    {
        DraftField.POSITIVE_CLASS,
        DraftField.DECISION_ACTOR_LABEL,
        DraftField.GUARDRAIL_THRESHOLD,
    }
)

# Al cambiar la clave, pierden su confirmación los campos dependientes.
DEPENDENT_FIELDS: dict[DraftField, frozenset[DraftField]] = {
    DraftField.TASK_TYPE: frozenset(
        {DraftField.POSITIVE_CLASS, DraftField.PRIMARY_METRIC, DraftField.METRIC_RATIONALE}
    ),
    DraftField.TARGET_COLUMN: frozenset({DraftField.POSITIVE_CLASS}),
    DraftField.PRIMARY_METRIC: frozenset(
        {
            DraftField.METRIC_RATIONALE,
            DraftField.GUARDRAIL_MODE,
            DraftField.GUARDRAIL_METRIC,
            DraftField.GUARDRAIL_THRESHOLD,
            DraftField.GUARDRAIL_RATIONALE,
        }
    ),
    DraftField.GUARDRAIL_MODE: frozenset(
        {
            DraftField.GUARDRAIL_METRIC,
            DraftField.GUARDRAIL_THRESHOLD,
            DraftField.GUARDRAIL_RATIONALE,
        }
    ),
    DraftField.GUARDRAIL_METRIC: frozenset(
        {DraftField.GUARDRAIL_THRESHOLD, DraftField.GUARDRAIL_RATIONALE}
    ),
}

_GUARDRAIL_DETAIL_FIELDS = (
    DraftField.GUARDRAIL_METRIC,
    DraftField.GUARDRAIL_THRESHOLD,
    DraftField.GUARDRAIL_RATIONALE,
)


class FieldSource(StrEnum):
    HUMAN = "HUMAN"
    DATASET = "DATASET"
    LLM = "LLM"
    FALLBACK = "FALLBACK"
    # GA-13: decisión confirmada antes en otro proyecto; aquí sólo se propone.
    SAVED_CONTEXT = "SAVED_CONTEXT"
    # V2-10: decisión adoptada en bloque desde las recomendaciones deterministas del núcleo.
    GOAL_SETTINGS_RECOMMENDATION = "GOAL_SETTINGS_RECOMMENDATION"


class ConfirmationStatus(StrEnum):
    EMPTY = "EMPTY"
    PROPOSED = "PROPOSED"
    CONFIRMED = "CONFIRMED"
    INVALIDATED = "INVALIDATED"


class Availability(StrEnum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class DraftUse(StrEnum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------------------
# Perfil seguro: lo único que sale del equipo
# --------------------------------------------------------------------------------------

Level = Literal["none", "low", "medium", "high"]
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
MAX_SAFE_COLUMN_NAME = 64


def sanitize_column_name(name: str) -> str:
    """Saneamiento sintáctico; la revisión de datos sensibles ocurre aparte, antes del envío."""

    cleaned = _CONTROL_CHARS.sub(" ", name).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:MAX_SAFE_COLUMN_NAME] or "(sin nombre)"


def _missing_level(rate: float) -> Level:
    if rate == 0:
        return "none"
    if rate < 0.05:
        return "low"
    if rate < 0.3:
        return "medium"
    return "high"


def _cardinality_level(unique_count: int, non_null: int) -> Level:
    if unique_count <= 2:
        return "none" if unique_count <= 1 else "low"
    ratio = unique_count / non_null if non_null else 0.0
    if unique_count <= 20:
        return "low"
    return "high" if ratio >= 0.5 else "medium"


class SafeColumnV1(StrictModel):
    ref: str = Field(pattern=r"^C\d{3}$")
    name: str = Field(min_length=1, max_length=MAX_SAFE_COLUMN_NAME)
    inferred_type: Literal["boolean", "numeric", "categorical", "string"]
    missing_level: Level
    cardinality_level: Level
    constant: bool
    possible_identifier: bool
    binary_candidate: bool


class SafeProfileForLLMV1(StrictModel):
    """Sin filas, valores, clases, min/max/media, nombres de archivo, hashes ni rutas."""

    schema_version: Literal["safe-profile.v1"] = "safe-profile.v1"
    size_band: Literal["small", "medium", "large"]
    columns: list[SafeColumnV1] = Field(min_length=1, max_length=100)

    @classmethod
    def from_profile(
        cls, profile: DatasetProfileV1, *, excluded_refs: set[str] | None = None
    ) -> "SafeProfileForLLMV1":
        excluded = excluded_refs or set()
        columns = []
        for index, column in enumerate(profile.columns, start=1):
            ref = f"C{index:03d}"
            if ref in excluded:
                continue
            non_null = profile.row_count - column.missing_count
            columns.append(
                SafeColumnV1(
                    ref=ref,
                    name=sanitize_column_name(column.name),
                    inferred_type=column.inferred_type,
                    missing_level=_missing_level(column.missing_rate),
                    cardinality_level=_cardinality_level(column.unique_count, non_null),
                    constant=column.constant,
                    possible_identifier=column.possible_identifier,
                    binary_candidate=is_binary_candidate(column),
                )
            )
        rows = profile.row_count
        band = "small" if rows < 1_000 else "medium" if rows < 20_000 else "large"
        return cls(size_band=band, columns=columns)


def is_binary_candidate(column: Any) -> bool:
    """Calculado localmente: exactamente dos valores y sin vacíos."""

    return column.unique_count == 2 and column.missing_count == 0


def column_refs(profile: DatasetProfileV1) -> dict[str, str]:
    """Referencia estable `C001` → nombre real de la columna."""

    return {f"C{index:03d}": column.name for index, column in enumerate(profile.columns, start=1)}


# --------------------------------------------------------------------------------------
# Salida del LLM: propone y pregunta; no confirma
# --------------------------------------------------------------------------------------

ProposableField = Literal[
    "objective",
    "target_column",
    "prediction_moment",
    "false_positive_consequence",
    "false_negative_consequence",
    "primary_metric",
    "metric_rationale",
    "guardrail_mode",
    "guardrail_metric",
    "guardrail_rationale",
]


class AssistantQuestion(StrictModel):
    question_id: str = Field(pattern=r"^[a-z0-9_-]{1,40}$")
    field: DraftField
    prompt: str = Field(min_length=5, max_length=500)
    why_needed: str = Field(min_length=5, max_length=500)


class AssistantProposal(StrictModel):
    """`value` es texto; para `target_column` es una referencia `C###`."""

    field: ProposableField
    value: str = Field(min_length=1, max_length=2000)
    explanation: str = Field(min_length=5, max_length=1000)


class AssistantFeatureNote(StrictModel):
    """Alerta sobre una columna; la decisión de uso y disponibilidad es siempre humana."""

    column_ref: str = Field(pattern=r"^C\d{3}$")
    suggested_use: Literal["include", "exclude"]
    possible_post_event: bool
    explanation: str = Field(min_length=5, max_length=500)


class AssistantTurnV1(StrictModel):
    schema_version: Literal["assistant-turn.v1"] = "assistant-turn.v1"
    assistant_message: str = Field(min_length=1, max_length=2000)
    questions: list[AssistantQuestion] = Field(max_length=3)
    proposals: list[AssistantProposal] = Field(max_length=12)
    feature_notes: list[AssistantFeatureNote] = Field(max_length=100)
    status: Literal["NEEDS_INPUT", "READY_FOR_REVIEW"]

    @model_validator(mode="after")
    def fields_are_unique(self) -> "AssistantTurnV1":
        fields = [item.field for item in self.proposals]
        if len(fields) != len(set(fields)):
            raise ValueError("Cada campo admite una sola propuesta por turno")
        refs = [item.column_ref for item in self.feature_notes]
        if len(refs) != len(set(refs)):
            raise ValueError("Cada columna admite una sola nota por turno")
        return self


# --------------------------------------------------------------------------------------
# Borrador con procedencia y confirmación por campo
# --------------------------------------------------------------------------------------


class DraftValue(StrictModel):
    value: str | int | float | bool | None = None
    source: FieldSource | None = None
    status: ConfirmationStatus = ConfirmationStatus.EMPTY
    explanation: str | None = Field(default=None, max_length=1000)
    confirmed_by: str | None = Field(default=None, max_length=200)
    confirmed_at: datetime | None = None


class FeatureDraft(StrictModel):
    column: str = Field(min_length=1, max_length=255)
    use: DraftUse = DraftUse.UNKNOWN
    available_at_prediction: Availability = Availability.UNKNOWN
    note: str | None = Field(default=None, max_length=1000)
    note_source: FieldSource | None = None
    possible_post_event: bool = False
    status: ConfirmationStatus = ConfirmationStatus.EMPTY
    confirmed_by: str | None = Field(default=None, max_length=200)
    confirmed_at: datetime | None = None

    @property
    def effective_use(self) -> FeatureUse:
        """«No» o «no sé» en uso o disponibilidad excluye la variable."""

        if self.use is DraftUse.INCLUDE and self.available_at_prediction is Availability.YES:
            return FeatureUse.INCLUDE
        return FeatureUse.EXCLUDE


class DraftNotReadyError(ValueError):
    """El borrador tiene campos sin confirmar; incluye la lista para mostrarla."""

    def __init__(self, missing: list[str]) -> None:
        super().__init__("Faltan confirmaciones: " + ", ".join(missing))
        self.missing = missing


def _now() -> datetime:
    return datetime.now(UTC)


class ContextDraftV1(StrictModel):
    schema_version: Literal["context-draft.v1"] = "context-draft.v1"
    project_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    revision: int = Field(default=0, ge=0)
    fields: dict[DraftField, DraftValue] = Field(
        default_factory=lambda: {name: DraftValue() for name in DraftField}
    )
    features: list[FeatureDraft] = Field(default_factory=list)

    @field_validator("fields")
    @classmethod
    def all_fields_present(cls, value: dict[DraftField, DraftValue]) -> dict:
        return {name: value.get(name, DraftValue()) for name in DraftField}

    # -- lectura ---------------------------------------------------------------------

    def get(self, name: DraftField) -> DraftValue:
        return self.fields[name]

    @property
    def task_type(self) -> str:
        return str(self.get(DraftField.TASK_TYPE).value or "binary_classification")

    def _required_fields(self) -> list[DraftField]:
        required = [
            name
            for name in DraftField
            if name not in _GUARDRAIL_DETAIL_FIELDS and name is not DraftField.TASK_TYPE
        ]
        if self.task_type != "binary_classification":
            required.remove(DraftField.POSITIVE_CLASS)
        if self.get(DraftField.GUARDRAIL_MODE).value == GuardrailMode.QUANTITATIVE.value:
            required.extend(_GUARDRAIL_DETAIL_FIELDS)
        return required

    def missing_confirmations(self) -> list[str]:
        missing = [
            name.value
            for name in self._required_fields()
            if self.get(name).status is not ConfirmationStatus.CONFIRMED
        ]
        candidates = self.candidate_features()
        if not candidates:
            missing.append("features")
        missing.extend(
            f"feature:{item.column}"
            for item in candidates
            if item.status is not ConfirmationStatus.CONFIRMED
        )
        return missing

    def candidate_features(self) -> list[FeatureDraft]:
        """Todas las columnas salvo el resultado a anticipar elegido."""

        target = self.get(DraftField.TARGET_COLUMN).value
        return [item for item in self.features if item.column != target]

    # -- escritura -------------------------------------------------------------------

    def _invalidate(self, names: frozenset[DraftField]) -> None:
        for name in names:
            current = self.get(name)
            if current.status is not ConfirmationStatus.EMPTY:
                self.fields[name] = current.model_copy(
                    update={
                        "status": ConfirmationStatus.INVALIDATED,
                        "confirmed_by": None,
                        "confirmed_at": None,
                    }
                )

    def _invalidate_features(self, *, availability_only: bool) -> None:
        for index, item in enumerate(self.features):
            if item.status is ConfirmationStatus.CONFIRMED:
                update: dict[str, Any] = {
                    "status": ConfirmationStatus.INVALIDATED,
                    "confirmed_by": None,
                    "confirmed_at": None,
                }
                if availability_only:
                    update["available_at_prediction"] = Availability.UNKNOWN
                self.features[index] = item.model_copy(update=update)

    def propose(
        self,
        name: DraftField,
        value: str | int | float | bool,
        *,
        source: FieldSource,
        explanation: str | None = None,
    ) -> None:
        """Registra una propuesta (IA, dataset o plantilla). Nunca confirma."""

        if source is FieldSource.HUMAN:
            raise ValueError("Una respuesta humana se registra con set_by_person")
        if source is FieldSource.LLM and name in HUMAN_ONLY_FIELDS:
            raise ValueError(f"La IA no puede proponer {name.value}")
        current = self.get(name)
        if current.status is ConfirmationStatus.CONFIRMED and current.value == value:
            return
        self._replace(name, value, source, ConfirmationStatus.PROPOSED, explanation, None)

    def set_by_person(
        self, name: DraftField, value: str | int | float | bool, *, actor_label: str
    ) -> None:
        """Valor escrito o elegido por la persona: queda confirmado con su etiqueta."""

        self._replace(
            name, value, FieldSource.HUMAN, ConfirmationStatus.CONFIRMED, None, actor_label
        )

    def confirm(self, name: DraftField, *, actor_label: str) -> None:
        """Adopta explícitamente la propuesta vigente de un campo concreto."""

        current = self.get(name)
        if current.value is None or current.status is not ConfirmationStatus.PROPOSED:
            raise ValueError(f"Sólo se puede confirmar una propuesta vigente en {name.value}")
        self.fields[name] = current.model_copy(
            update={
                "status": ConfirmationStatus.CONFIRMED,
                "confirmed_by": actor_label,
                "confirmed_at": _now(),
            }
        )
        self.revision += 1

    def _replace(
        self,
        name: DraftField,
        value: str | int | float | bool,
        source: FieldSource,
        status: ConfirmationStatus,
        explanation: str | None,
        actor_label: str | None,
    ) -> None:
        previous = self.get(name).value
        self.fields[name] = DraftValue(
            value=value,
            source=source,
            status=status,
            explanation=explanation,
            confirmed_by=actor_label,
            confirmed_at=_now() if actor_label else None,
        )
        if previous is not None and previous != value:
            self._invalidate(DEPENDENT_FIELDS.get(name, frozenset()))
            if name is DraftField.PREDICTION_MOMENT:
                self._invalidate_features(availability_only=True)
        if name is DraftField.TARGET_COLUMN and previous is not None and previous != value:
            self._invalidate_features(availability_only=False)
        self.revision += 1

    def set_feature(
        self,
        column: str,
        *,
        use: DraftUse,
        available_at_prediction: Availability,
        actor_label: str,
        source: FieldSource | None = None,
        explanation: str | None = None,
    ) -> None:
        """Decisión humana explícita sobre una variable."""

        for index, item in enumerate(self.features):
            if item.column == column:
                trace = {}
                if source is not None:
                    trace = {"note": explanation, "note_source": source}
                self.features[index] = item.model_copy(
                    update={
                        "use": use,
                        "available_at_prediction": available_at_prediction,
                        "status": ConfirmationStatus.CONFIRMED,
                        "confirmed_by": actor_label,
                        "confirmed_at": _now(),
                        **trace,
                    }
                )
                self.revision += 1
                return
        raise ValueError(f"Columna desconocida en el borrador: {column}")

    def unconfirm_feature_recommendation(self, column: str, *, actor_label: str) -> None:
        """Deshace una recomendación aceptada sin tocar una decisión manual posterior."""

        if not actor_label.strip():
            raise ValueError("La persona que deshace la recomendación debe estar identificada")
        for index, item in enumerate(self.features):
            if item.column != column:
                continue
            if (
                item.status is ConfirmationStatus.CONFIRMED
                and item.note_source is FieldSource.GOAL_SETTINGS_RECOMMENDATION
            ):
                self.features[index] = item.model_copy(
                    update={
                        "status": ConfirmationStatus.PROPOSED,
                        "confirmed_by": None,
                        "confirmed_at": None,
                    }
                )
                self.revision += 1
            return
        raise ValueError(f"Columna desconocida en el borrador: {column}")

    # -- conversión ------------------------------------------------------------------

    def to_business_context(self) -> BusinessContextV1:
        """Sólo con todo confirmado. `BusinessContextService` revalida contra los datos."""

        missing = self.missing_confirmations()
        if missing:
            raise DraftNotReadyError(missing)
        value = {name: self.get(name).value for name in DraftField}
        rules = []
        if value[DraftField.GUARDRAIL_MODE] == GuardrailMode.QUANTITATIVE.value:
            rules.append(
                {
                    "metric": value[DraftField.GUARDRAIL_METRIC],
                    "operator": ">=",
                    "threshold": float(value[DraftField.GUARDRAIL_THRESHOLD]),
                    "rationale": value[DraftField.GUARDRAIL_RATIONALE],
                }
            )
        return BusinessContextV1.model_validate(
            {
                "objective": value[DraftField.OBJECTIVE],
                "target_column": value[DraftField.TARGET_COLUMN],
                "task_type": self.task_type,
                "positive_class": (
                    value[DraftField.POSITIVE_CLASS]
                    if self.task_type == "binary_classification"
                    else None
                ),
                "prediction_moment": value[DraftField.PREDICTION_MOMENT],
                "false_positive_consequence": value[DraftField.FALSE_POSITIVE_CONSEQUENCE],
                "false_negative_consequence": value[DraftField.FALSE_NEGATIVE_CONSEQUENCE],
                "primary_metric": value[DraftField.PRIMARY_METRIC],
                "metric_rationale": value[DraftField.METRIC_RATIONALE],
                "decision_actor_label": value[DraftField.DECISION_ACTOR_LABEL],
                "feature_decisions": [
                    {
                        "column": item.column,
                        "use": item.effective_use.value,
                        "available_at_prediction": (
                            item.available_at_prediction is Availability.YES
                        ),
                        "reason": _feature_reason(item),
                    }
                    for item in self.candidate_features()
                ],
                "guardrails": {"mode": value[DraftField.GUARDRAIL_MODE], "rules": rules},
            }
        )

    def seed_from_context(self, context: BusinessContextV1) -> None:
        """Parte de las decisiones vigentes para editarlas; quedan confirmadas por su autor."""

        actor = context.decision_actor_label
        values: dict[DraftField, Any] = {
            DraftField.DECISION_ACTOR_LABEL: actor,
            DraftField.OBJECTIVE: context.objective,
            DraftField.TASK_TYPE: context.task_type,
            DraftField.TARGET_COLUMN: context.target_column,
            DraftField.POSITIVE_CLASS: context.positive_class,
            DraftField.PREDICTION_MOMENT: context.prediction_moment,
            DraftField.FALSE_POSITIVE_CONSEQUENCE: context.false_positive_consequence,
            DraftField.FALSE_NEGATIVE_CONSEQUENCE: context.false_negative_consequence,
            DraftField.PRIMARY_METRIC: context.primary_metric.value,
            DraftField.METRIC_RATIONALE: context.metric_rationale,
            DraftField.GUARDRAIL_MODE: context.guardrails.mode.value,
        }
        if context.guardrails.rules:
            rule = context.guardrails.rules[0]
            values[DraftField.GUARDRAIL_METRIC] = rule.metric.value
            values[DraftField.GUARDRAIL_THRESHOLD] = rule.threshold
            values[DraftField.GUARDRAIL_RATIONALE] = rule.rationale
        for name, value in values.items():
            if value is not None:
                self.set_by_person(name, value, actor_label=actor)
        decisions = {item.column: item for item in context.feature_decisions}
        for item in list(self.features):
            decision = decisions.get(item.column)
            if decision is None:
                continue
            self.set_feature(
                item.column,
                use=DraftUse(decision.use.value),
                available_at_prediction=(
                    Availability.YES if decision.available_at_prediction else Availability.NO
                ),
                actor_label=actor,
            )

    def reuse_context(self, context: BusinessContextV1, *, origin: str) -> int:
        """GA-13: propone las decisiones de un contexto guardado; nunca confirma.

        Sólo se proponen campos vacíos o aún propuestos, y columnas que existen en este
        archivo. El responsable no se copia: cada persona lo declara por sí misma.
        """

        explanation = f"Así se decidió en «{origin}». Confírmalo o cámbialo."
        columns = {item.column for item in self.features}
        values: dict[DraftField, Any] = {
            DraftField.OBJECTIVE: context.objective,
            DraftField.PREDICTION_MOMENT: context.prediction_moment,
            DraftField.FALSE_POSITIVE_CONSEQUENCE: context.false_positive_consequence,
            DraftField.FALSE_NEGATIVE_CONSEQUENCE: context.false_negative_consequence,
            DraftField.PRIMARY_METRIC: context.primary_metric.value,
            DraftField.METRIC_RATIONALE: context.metric_rationale,
            DraftField.GUARDRAIL_MODE: context.guardrails.mode.value,
        }
        if context.target_column in columns:
            values[DraftField.TASK_TYPE] = context.task_type
            values[DraftField.TARGET_COLUMN] = context.target_column
            values[DraftField.POSITIVE_CLASS] = context.positive_class
        if context.guardrails.rules:
            rule = context.guardrails.rules[0]
            values[DraftField.GUARDRAIL_METRIC] = rule.metric.value
            values[DraftField.GUARDRAIL_THRESHOLD] = rule.threshold
            values[DraftField.GUARDRAIL_RATIONALE] = rule.rationale
        proposed = 0
        for name, value in values.items():
            if self.get(name).status is ConfirmationStatus.CONFIRMED:
                continue
            self.propose(name, value, source=FieldSource.SAVED_CONTEXT, explanation=explanation)
            proposed += 1
        decisions = {item.column: item for item in context.feature_decisions}
        for index, item in enumerate(self.features):
            decision = decisions.get(item.column)
            if decision is None or item.status is ConfirmationStatus.CONFIRMED:
                continue
            self.features[index] = item.model_copy(
                update={
                    "use": DraftUse(decision.use.value),
                    "available_at_prediction": (
                        Availability.YES if decision.available_at_prediction else Availability.NO
                    ),
                    "status": ConfirmationStatus.PROPOSED,
                    "note": f"En «{origin}»: {decision.reason}"[:1000],
                    "note_source": FieldSource.SAVED_CONTEXT,
                }
            )
            proposed += 1
        self.revision += 1
        return proposed

    @classmethod
    def start(
        cls,
        *,
        project_id: str,
        dataset_id: str,
        dataset_sha256: str,
        profile: DatasetProfileV1,
    ) -> "ContextDraftV1":
        """Borrador vacío: identificadores y constantes quedan propuestos para excluir."""

        features = [
            FeatureDraft(
                column=column.name,
                use=(
                    DraftUse.EXCLUDE
                    if column.possible_identifier or column.constant
                    else DraftUse.UNKNOWN
                ),
                note=(
                    "Parece un identificador o un valor constante; no ayuda a predecir."
                    if column.possible_identifier or column.constant
                    else None
                ),
                note_source=(
                    FieldSource.DATASET if column.possible_identifier or column.constant else None
                ),
                status=(
                    ConfirmationStatus.PROPOSED
                    if column.possible_identifier or column.constant
                    else ConfirmationStatus.EMPTY
                ),
            )
            for column in profile.columns
        ]
        return cls(
            project_id=project_id,
            dataset_id=dataset_id,
            dataset_sha256=dataset_sha256,
            features=features,
        )


def _feature_reason(item: FeatureDraft) -> str:
    if item.available_at_prediction is not Availability.YES:
        return "Declarada no disponible, o sin confirmar, en el momento de predicción"
    if item.use is DraftUse.INCLUDE:
        return "Disponible al predecir e incluida por decisión humana"
    return "Excluida por decisión humana"

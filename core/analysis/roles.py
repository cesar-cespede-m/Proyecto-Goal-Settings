"""Inferencia genérica del papel de cada columna a partir de su comportamiento."""

from __future__ import annotations

from statistics import median
from typing import Literal

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.data_engine.models import DatasetProfileV1

type ColumnRoleName = Literal[
    "entity",
    "moment",
    "stage",
    "event",
    "measure",
    "attribute",
    "outcome_candidate",
    "other",
]

_ALLOWED_ROLES = {
    "entity",
    "moment",
    "stage",
    "event",
    "measure",
    "attribute",
    "outcome_candidate",
    "other",
}
_MAX_STAGE_VALUES = 15
_MIN_STAGE_SUPPORT = 0.8
_FIXED_ATTRIBUTE_SHARE = 0.95


class StrictRoleModel(BaseModel):
    """Contrato sin campos implícitos ni coerciones silenciosas."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )


class ColumnRoleConfirmationV1(StrictRoleModel):
    role: ColumnRoleName
    actor_label: str = Field(min_length=1, max_length=200)


class ColumnRoleV1(StrictRoleModel):
    column: str = Field(min_length=1, max_length=255)
    role: ColumnRoleName
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=500)
    stage_order: list[str] = Field(default_factory=list, max_length=_MAX_STAGE_VALUES)
    confirmation: ColumnRoleConfirmationV1 | None = None

    @model_validator(mode="after")
    def stage_has_an_order(self) -> ColumnRoleV1:
        if self.role == "stage" and len(self.stage_order) < 2:
            raise ValueError("Una etapa requiere al menos dos valores ordenados")
        if self.role != "stage" and self.stage_order:
            raise ValueError("Sólo una etapa puede declarar un orden")
        if len(self.stage_order) != len(set(self.stage_order)):
            raise ValueError("El orden de etapas no puede repetir valores")
        return self


class ColumnRolesV1(StrictRoleModel):
    schema_version: Literal["column-roles.v1"] = "column-roles.v1"
    items: list[ColumnRoleV1] = Field(min_length=1)

    @model_validator(mode="after")
    def columns_are_unique(self) -> ColumnRolesV1:
        columns = [item.column for item in self.items]
        if len(columns) != len(set(columns)):
            raise ValueError("Cada columna debe aparecer una sola vez")
        return self

    def with_confirmation(
        self,
        column: str,
        role: ColumnRoleName | str,
        actor_label: str,
    ) -> ColumnRolesV1:
        """Devuelve una copia confirmada; la persistencia pertenece a otra historia."""

        role_value = str(role).strip()
        if role_value not in _ALLOWED_ROLES:
            raise ValueError(f"Papel no permitido: {role_value}")
        confirmation = ColumnRoleConfirmationV1(
            role=role_value,  # type: ignore[arg-type]
            actor_label=actor_label,
        )
        found = False
        items: list[ColumnRoleV1] = []
        for item in self.items:
            if item.column == column:
                found = True
                items.append(item.model_copy(update={"confirmation": confirmation}))
            else:
                items.append(item)
        if not found:
            raise ValueError(f"La columna no existe en la propuesta: {column}")
        return ColumnRolesV1(items=items)


def infer_roles(frame: pd.DataFrame, profile: DatasetProfileV1) -> ColumnRolesV1:
    """Propone papeles por repetición, tipo, estabilidad y orden temporal observados."""

    _validate_inputs(frame, profile)
    moments = {
        column: evidence
        for column in frame.columns
        if (evidence := _datetime_evidence(frame[column])) is not None
    }
    entities = {
        column: evidence
        for column in frame.columns
        if column not in moments and (evidence := _entity_evidence(frame[column])) is not None
    }
    primary_entity = max(entities, key=lambda name: entities[name][0], default=None)
    primary_moment = max(moments, key=lambda name: moments[name][0], default=None)

    stage_evidence: dict[str, tuple[list[str], float, float]] = {}
    if primary_entity is not None and primary_moment is not None:
        for column in frame.columns:
            if column in moments or column in entities or _is_numeric(frame[column]):
                continue
            unique_count = int(frame[column].nunique(dropna=True))
            if 2 <= unique_count <= _MAX_STAGE_VALUES:
                evidence = _ordered_category_evidence(
                    frame,
                    entity=primary_entity,
                    moment=moments[primary_moment][1],
                    category=column,
                )
                order, support, transition_share = evidence
                if support >= _MIN_STAGE_SUPPORT:
                    stage_evidence[column] = (order, support, transition_share)

    items = [
        _role_for_column(
            frame,
            column,
            moments=moments,
            entities=entities,
            primary_entity=primary_entity,
            primary_moment=primary_moment,
            stage_evidence=stage_evidence,
        )
        for column in frame.columns
    ]
    return ColumnRolesV1(items=items)


def _validate_inputs(frame: pd.DataFrame, profile: DatasetProfileV1) -> None:
    if len(frame) != profile.row_count:
        raise ValueError("El perfil y la tabla no tienen la misma cantidad de filas")
    frame_columns = [str(column) for column in frame.columns]
    profile_columns = [column.name for column in profile.columns]
    if len(frame_columns) != len(set(frame_columns)):
        raise ValueError("La tabla contiene nombres de columna repetidos")
    if set(frame_columns) != set(profile_columns) or len(frame_columns) != profile.column_count:
        raise ValueError("El perfil y la tabla no describen las mismas columnas")


def _datetime_evidence(series: pd.Series) -> tuple[float, pd.Series] | None:
    values = series.dropna()
    if values.empty or values.nunique(dropna=True) < 2:
        return None
    if is_datetime64_any_dtype(series.dtype):
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    elif _is_numeric(series) or is_bool_dtype(series.dtype):
        return None
    else:
        parsed = pd.to_datetime(series.astype("string"), errors="coerce", format="mixed", utc=True)
    coverage = float(parsed.notna().sum() / series.notna().sum())
    if coverage < _MIN_STAGE_SUPPORT:
        return None
    return round(coverage, 6), parsed


def _entity_evidence(series: pd.Series) -> tuple[float, float] | None:
    if _is_numeric(series) or is_bool_dtype(series.dtype):
        return None
    values = series.dropna()
    counts = values.value_counts(dropna=True)
    if len(counts) <= _MAX_STAGE_VALUES:
        return None
    median_rows = float(counts.median()) if not counts.empty else 0.0
    if median_rows < 2:
        return None
    coverage = float(len(values) / len(series))
    confidence = min(1.0, 0.8 + min(median_rows - 2, 2) * 0.075 + coverage * 0.05)
    return round(confidence, 6), median_rows


def _ordered_category_evidence(
    frame: pd.DataFrame,
    *,
    entity: str,
    moment: pd.Series,
    category: str,
) -> tuple[list[str], float, float]:
    work = pd.DataFrame(
        {
            "entity": frame[entity],
            "moment": moment,
            "category": frame[category],
            "row": range(len(frame)),
        }
    ).dropna(subset=["entity", "moment", "category"])
    work = work.sort_values(["entity", "moment", "row"], kind="stable")
    sequences: list[list[object]] = []
    first_seen: dict[object, int] = {}
    positions: dict[object, list[int]] = {}
    for _, history in work.groupby("entity", sort=False):
        sequence: list[object] = []
        for value, row in zip(history["category"], history["row"], strict=True):
            if not sequence or value != sequence[-1]:
                sequence.append(value)
            first_seen[value] = min(first_seen.get(value, int(row)), int(row))
        sequences.append(sequence)
        for position, value in enumerate(sequence):
            positions.setdefault(value, []).append(position)
    if not sequences:
        return [], 0.0, 0.0

    ordered_values = sorted(
        positions,
        key=lambda value: (
            median(positions[value]),
            first_seen[value],
            _human_value(value),
        ),
    )
    ranks = {value: index for index, value in enumerate(ordered_values)}
    changing_sequences = [sequence for sequence in sequences if len(set(sequence)) >= 2]
    consistent = sum(
        all(
            ranks[left] <= ranks[right] for left, right in zip(sequence, sequence[1:], strict=False)
        )
        for sequence in changing_sequences
    )
    transitions = len(changing_sequences)
    return (
        [str(value) for value in ordered_values],
        consistent / transitions if transitions else 0.0,
        transitions / len(sequences),
    )


def _role_for_column(
    frame: pd.DataFrame,
    column: str,
    *,
    moments: dict[str, tuple[float, pd.Series]],
    entities: dict[str, tuple[float, float]],
    primary_entity: str | None,
    primary_moment: str | None,
    stage_evidence: dict[str, tuple[list[str], float, float]],
) -> ColumnRoleV1:
    series = frame[column]
    unique_count = int(series.nunique(dropna=True))
    if column in moments:
        confidence = moments[column][0]
        return _item(
            column,
            "moment",
            confidence,
            f"El {_percent(confidence)} de sus valores no vacíos se interpreta como fecha u hora.",
        )
    if column in entities:
        confidence, median_rows = entities[column]
        return _item(
            column,
            "entity",
            confidence,
            "Cada valor aparece en una mediana de "
            f"{_plain_number(median_rows)} filas, por lo que permite agrupar varias filas "
            "del mismo caso.",
        )
    if column in stage_evidence:
        order, support, _ = stage_evidence[column]
        readable_order = " → ".join(_human_value(value) for value in order)
        return _item(
            column,
            "stage",
            support,
            f"En el {_percent(support)} de las entidades que cambian de valor, éstos aparecen "
            f"siempre en el mismo orden: {readable_order}.",
            stage_order=order,
        )

    fixed_share = _fixed_share(frame, primary_entity, column)
    if unique_count == 2 and (primary_entity is None or fixed_share >= _FIXED_ATTRIBUTE_SHARE):
        if primary_entity is None:
            reason = (
                "Tiene dos valores posibles en una tabla con una fila por caso, por lo que "
                "puede ser un resultado que la persona debe confirmar."
            )
            confidence = 0.8
        else:
            reason = (
                f"Tiene dos valores posibles y se mantiene fijo en el {_percent(fixed_share)} "
                "de las entidades, por lo que puede ser un resultado que la persona debe "
                "confirmar."
            )
            confidence = fixed_share
        return _item(column, "outcome_candidate", confidence, reason)

    if _is_numeric(series):
        return _item(
            column,
            "measure",
            0.9,
            "Contiene números que pueden resumirse, compararse o seguirse como una medida.",
        )

    if 2 <= unique_count <= _MAX_STAGE_VALUES:
        if primary_entity is None:
            return _item(
                column,
                "attribute",
                0.75,
                f"Tiene {unique_count} valores posibles en una tabla con una fila por caso.",
            )
        if fixed_share >= _FIXED_ATTRIBUTE_SHARE:
            return _item(
                column,
                "attribute",
                fixed_share,
                f"Conserva un solo valor en el {_percent(fixed_share)} de las entidades.",
            )
        if primary_moment is not None:
            order_support = _category_order_support(
                frame,
                entity=primary_entity,
                moment=moments[primary_moment][1],
                category=column,
            )
            return _item(
                column,
                "event",
                max(0.6, 1 - order_support),
                "Cambia entre filas y sólo el "
                f"{_percent(order_support)} de las entidades que cambian de valor mantiene "
                "un orden común, por lo que describe lo que ocurre en cada registro.",
            )

    if unique_count == 1:
        return _item(
            column,
            "other",
            0.9,
            "Mantiene el mismo valor en todas las filas y no aporta una diferencia observable.",
        )
    return _item(
        column,
        "other",
        0.5,
        "Su comportamiento no aporta evidencia suficiente para asignarle otro papel.",
    )


def _category_order_support(
    frame: pd.DataFrame,
    *,
    entity: str,
    moment: pd.Series,
    category: str,
) -> float:
    return _ordered_category_evidence(
        frame,
        entity=entity,
        moment=moment,
        category=category,
    )[1]


def _fixed_share(frame: pd.DataFrame, entity: str | None, column: str) -> float:
    if entity is None:
        return 1.0
    valid = frame[[entity, column]].dropna()
    if valid.empty:
        return 0.0
    counts = valid.groupby(entity, sort=False)[column].nunique(dropna=True)
    return float((counts <= 1).mean()) if not counts.empty else 0.0


def _item(
    column: str,
    role: ColumnRoleName,
    confidence: float,
    reason: str,
    *,
    stage_order: list[str] | None = None,
) -> ColumnRoleV1:
    return ColumnRoleV1(
        column=str(column),
        role=role,
        confidence=round(float(confidence), 6),
        reason=reason,
        stage_order=stage_order or [],
    )


def _is_numeric(series: pd.Series) -> bool:
    return is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype)


def _percent(value: float) -> str:
    return f"{round(value * 100)} %"


def _plain_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.1f}"


def _human_value(value: object) -> str:
    return str(value).casefold().replace("_", " ").replace("-", " ")

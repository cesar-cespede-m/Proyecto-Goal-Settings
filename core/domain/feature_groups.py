"""V2-10: recomendaciones deterministas y reversibles para columnas del modelo."""

from enum import StrEnum
from typing import Literal

from pydantic import Field

from core.data_engine.models import ColumnProfile, DatasetProfileV1, OutcomeAssociationV1
from core.domain.assistant import (
    Availability,
    ConfirmationStatus,
    ContextDraftV1,
    DraftUse,
    FeatureDraft,
    FieldSource,
)
from core.domain.context import StrictModel


class FeatureGroup(StrEnum):
    USE = "use"
    EXCLUDE = "exclude"
    ASK = "ask"


class FeatureGroupItemV1(StrictModel):
    column: str = Field(min_length=1, max_length=255)
    group: FeatureGroup
    reason: str = Field(min_length=1, max_length=500)
    confirmed: bool


class FeatureGroupsV1(StrictModel):
    schema_version: Literal["feature-groups.v1"] = "feature-groups.v1"
    items: list[FeatureGroupItemV1]

    def in_group(self, group: FeatureGroup) -> list[FeatureGroupItemV1]:
        return [item for item in self.items if item.group is group]


def group_features(
    draft: ContextDraftV1,
    profile: DatasetProfileV1,
    *,
    target_column: str,
) -> FeatureGroupsV1:
    """Agrupa cada candidata con señales locales, agregadas y explicables."""

    profiles = {column.name: column for column in profile.columns}
    if target_column not in profiles:
        raise ValueError(f"La columna de resultado no existe en el perfil: {target_column}")
    associations = {
        item.feature_column: item
        for item in profile.outcome_associations
        if item.target_column == target_column
    }
    items = []
    for feature in draft.features:
        if feature.column == target_column:
            continue
        column = profiles.get(feature.column)
        if column is None:
            continue
        group, reason = _recommendation(feature, column, associations.get(feature.column))
        if feature.status is ConfirmationStatus.CONFIRMED:
            group = (
                FeatureGroup.USE
                if feature.use is DraftUse.INCLUDE
                and feature.available_at_prediction is Availability.YES
                else FeatureGroup.EXCLUDE
            )
        items.append(
            FeatureGroupItemV1(
                column=feature.column,
                group=group,
                reason=reason,
                confirmed=feature.status is ConfirmationStatus.CONFIRMED,
            )
        )
    return FeatureGroupsV1(items=items)


def accept_recommendations(
    draft: ContextDraftV1,
    groups: FeatureGroupsV1,
    *,
    actor_label: str,
) -> int:
    """Confirma en bloque sólo recomendaciones seguras; nunca responde las dudas."""

    accepted = 0
    current = {item.column: item for item in draft.features}
    for item in groups.items:
        feature = current.get(item.column)
        if (
            feature is None
            or feature.status is ConfirmationStatus.CONFIRMED
            or item.group is FeatureGroup.ASK
        ):
            continue
        use = DraftUse.INCLUDE if item.group is FeatureGroup.USE else DraftUse.EXCLUDE
        availability = Availability.YES if item.group is FeatureGroup.USE else Availability.UNKNOWN
        draft.set_feature(
            item.column,
            use=use,
            available_at_prediction=availability,
            actor_label=actor_label,
            source=FieldSource.GOAL_SETTINGS_RECOMMENDATION,
            explanation=f"Recomendación de Goal Settings: {item.reason}",
        )
        accepted += 1
    return accepted


def undo_recommendations(
    draft: ContextDraftV1,
    columns: list[str],
    *,
    actor_label: str,
) -> None:
    """Vuelve a dejar pendientes recomendaciones aceptadas, sin deshacer decisiones manuales."""

    for column in dict.fromkeys(columns):
        draft.unconfirm_feature_recommendation(column, actor_label=actor_label)


def _recommendation(
    feature: FeatureDraft,
    column: ColumnProfile,
    association: OutcomeAssociationV1 | None,
) -> tuple[FeatureGroup, str]:
    if column.semantic_type == "personal":
        return (
            FeatureGroup.EXCLUDE,
            "Conviene excluirla porque podría contener datos personales.",
        )
    if column.constant:
        return (
            FeatureGroup.EXCLUDE,
            "Conviene excluirla porque tiene el mismo valor en todas las filas.",
        )
    if (
        column.possible_identifier
        or column.high_cardinality
        or column.semantic_type == "identifier"
    ):
        return (
            FeatureGroup.EXCLUDE,
            "Conviene excluirla porque parece un identificador o es casi única por fila.",
        )
    if column.missing_rate >= 0.3:
        return (
            FeatureGroup.EXCLUDE,
            "Conviene excluirla porque faltan datos en al menos el 30 % de las filas.",
        )
    if association is not None:
        strong_categorical = (
            association.signal == "categorical_separation"
            and association.strength > 0.95
            and association.coverage > 0.9
        )
        strong_numeric = association.signal == "numeric_correlation" and association.strength > 0.9
        if strong_categorical or strong_numeric:
            return (
                FeatureGroup.EXCLUDE,
                "Podría conocerse recién después del resultado: confírmalo; en este archivo "
                "separa el resultado casi perfectamente.",
            )
        return (
            FeatureGroup.ASK,
            "Podría conocerse recién después del resultado: confírmalo antes de usarla.",
        )
    if feature.possible_post_event:
        return (
            FeatureGroup.ASK,
            "Podría conocerse recién después del resultado: confírmalo antes de usarla.",
        )
    return (
        FeatureGroup.USE,
        "No aparecen señales en contra; se recomienda usarla.",
    )

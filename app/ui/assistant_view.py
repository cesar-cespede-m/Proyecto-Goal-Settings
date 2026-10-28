"""Pantalla del asistente de contexto (UX-01.9): conversación, vista previa y fichas.

La IA sólo propone. Cada decisión se confirma campo por campo; no existe «confirmar todo».
`core/` valida y guarda; esta vista sólo traduce acciones de la persona.
"""

import json
import re
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any

import streamlit as st

from app.ui.components import render_field_card
from app.ui.copy import METRIC_LABELS
from app.ui.messages import fallback_reason_text, render_error
from core.ai_engine import LLMProvider
from core.ai_engine.context_assistant import ContextAssistant, apply_turn
from core.ai_engine.outgoing_review import review_outgoing
from core.data_engine.models import DatasetProfileV1
from core.db.repositories import WorkflowRepository
from core.domain.assistant import (
    Availability,
    ConfirmationStatus,
    DraftField,
    DraftNotReadyError,
    DraftUse,
    FieldSource,
    SafeProfileForLLMV1,
    column_refs,
    is_binary_candidate,
)
from core.domain.context_service import BusinessContextService
from core.domain.draft_service import ContextDraftService, StoredDraft
from core.domain.errors import ContextValidationError, InvalidStateTransitionError
from core.domain.feature_groups import (
    FeatureGroup,
    accept_recommendations,
    group_features,
    undo_recommendations,
)
from core.domain.recipe_bundle import RecipeBundleError

SOURCE_LABELS = {
    FieldSource.HUMAN: "Lo dijiste tú",
    FieldSource.DATASET: "Calculado del archivo",
    FieldSource.LLM: "Sugerido por IA",
    FieldSource.FALLBACK: "Plantilla sin IA",
    FieldSource.SAVED_CONTEXT: "De un contexto guardado",
}
STATUS_LABELS = {
    ConfirmationStatus.EMPTY: "Falta responder",
    ConfirmationStatus.PROPOSED: "Propuesto",
    ConfirmationStatus.CONFIRMED: "Confirmado",
    ConfirmationStatus.INVALIDATED: "Debe revisarse",
}
FIELD_LABELS = {
    DraftField.OBJECTIVE: ("objective", "Decisión que quieres apoyar"),
    DraftField.TASK_TYPE: ("task_type", "Qué forma tiene lo que quieres anticipar"),
    DraftField.TARGET_COLUMN: ("target_column", "Resultado que quieres anticipar"),
    DraftField.POSITIVE_CLASS: ("positive_class", "Evento que quieres detectar"),
    DraftField.PREDICTION_MOMENT: ("prediction_moment", "Cuándo usarías el aviso"),
    DraftField.FALSE_POSITIVE_CONSEQUENCE: (
        "false_positive",
        "Si el sistema avisa y no pasa nada",
    ),
    DraftField.FALSE_NEGATIVE_CONSEQUENCE: (
        "false_negative",
        "Si el sistema no avisa y el problema ocurre",
    ),
    DraftField.PRIMARY_METRIC: ("primary_metric", "Qué te importa más"),
    DraftField.METRIC_RATIONALE: ("metric_rationale", "Por qué elegiste eso"),
    DraftField.DECISION_ACTOR_LABEL: ("decision_actor", "Responsable declarado"),
    DraftField.GUARDRAIL_MODE: ("guardrail_mode", "Condición mínima"),
    DraftField.GUARDRAIL_METRIC: ("guardrail_metric", "Medida de la condición mínima"),
    DraftField.GUARDRAIL_THRESHOLD: ("guardrail_threshold", "Valor mínimo"),
    DraftField.GUARDRAIL_RATIONALE: ("guardrail_rationale", "Razón de la condición mínima"),
}
TEXT_FIELDS = (
    DraftField.OBJECTIVE,
    DraftField.PREDICTION_MOMENT,
    DraftField.FALSE_POSITIVE_CONSEQUENCE,
    DraftField.FALSE_NEGATIVE_CONSEQUENCE,
    DraftField.METRIC_RATIONALE,
)
METRIC_QUESTION = {
    "recall": "No dejar pasar ningún caso real, aunque haya avisos de más",
    "precision": "Que cuando avise, acierte, aunque se escape alguno",
    "f1": "Un equilibrio entre las dos anteriores",
    "balanced_accuracy": (
        "Acertar tanto cuando ocurre como cuando no ocurre, aunque ocurra pocas veces"
    ),
    "average_precision": (
        "Armar una lista de prioridades: que los primeros de la lista sean los que de verdad "
        "terminan ocurriendo"
    ),
    "roc_auc": "Que, en general, los casos que ocurren reciban más puntaje que los que no",
    "macro_f1": "Acertar parejo en todas las categorías, también en las poco frecuentes",
    "mae": "Equivocarme lo menos posible en promedio",
    "rmse": "Evitar sobre todo las equivocaciones grandes",
}
TASK_TYPES = {
    "binary_classification": "Una respuesta de dos opciones (por ejemplo, sí o no)",
    "multiclass_classification": "Una entre varias categorías",
    "regression": "Un número (una cantidad, un valor, un tiempo)",
}
METRICS_BY_TASK_TYPE = {
    "binary_classification": [
        "recall",
        "precision",
        "f1",
        "balanced_accuracy",
        "average_precision",
        "roc_auc",
    ],
    "multiclass_classification": ["macro_f1", "balanced_accuracy"],
    "regression": ["mae", "rmse"],
}


def target_candidates(profile: Any, task_type: str) -> list[str]:
    """Columnas que pueden ser el resultado, según el tipo elegido."""

    if task_type == "regression":
        return [
            column.name
            for column in profile.columns
            if column.inferred_type == "numeric"
            and column.missing_count == 0
            and column.unique_count >= 3
        ]
    if task_type == "multiclass_classification":
        return [
            column.name
            for column in profile.columns
            if 3 <= column.unique_count <= 20 and column.missing_count == 0
        ]
    return [column.name for column in profile.columns if is_binary_candidate(column)]


GUARDRAIL_MODES = {
    "no_quantitative_guardrail": "No exigir una condición mínima adicional",
    "quantitative": "Exigir una condición mínima",
}
TYPE_LABELS = {
    "numeric": "número",
    "categorical": "categoría",
    "string": "texto",
    "boolean": "sí/no",
    "datetime": "fecha",
}
_COLUMN_REF = re.compile(r"\bC\d{3}\b")


def with_column_names(text: str, refs: dict[str, str]) -> str:
    """UX-17: la IA trabaja con referencias C001…; la persona ve el nombre de la columna."""

    return _COLUMN_REF.sub(
        lambda match: f"«{refs[match.group(0)]}»" if match.group(0) in refs else match.group(0),
        text,
    )


def _commit(service: ContextDraftService, stored: StoredDraft) -> bool:
    try:
        service.save(stored)
    except InvalidStateTransitionError as error:
        render_error(error, fallback_code=str(error))
        return False
    st.rerun()
    return True  # pragma: no cover - st.rerun interrumpe la ejecución


def _status_line(stored: StoredDraft, field: DraftField, *, has_default: bool = False) -> None:
    item = stored.draft.get(field)
    if has_default and item.status is ConfirmationStatus.EMPTY:
        st.caption("Por defecto: se deja así si no la cambias.")
        return
    parts = [f"Estado: {STATUS_LABELS[item.status]}"]
    if item.source is not None:
        parts.append(f"Origen: {SOURCE_LABELS[item.source]}")
    if item.confirmed_by:
        parts.append(f"Confirmó: {item.confirmed_by}")
    st.caption(" · ".join(parts))
    if item.explanation and item.source is FieldSource.LLM:
        st.info(f"Por qué lo sugiere la IA: {item.explanation}")
    elif item.explanation and item.source is FieldSource.SAVED_CONTEXT:
        st.info(item.explanation)


def _accept_button(
    service: ContextDraftService, stored: StoredDraft, field: DraftField, actor: str
) -> None:
    item = stored.draft.get(field)
    if item.status is ConfirmationStatus.PROPOSED and item.value is not None:
        if st.button(
            "Aceptar la propuesta de la IA", key=f"accept-{stored.draft_id}-{field.value}"
        ):
            stored.draft.confirm(field, actor_label=actor)
            _commit(service, stored)


ACTOR_MAX_CHARS = 60
_ACTOR_MAX_WORDS = 6
_GOAL_START = re.compile(
    r"^(quiero|quisiera|necesito|tengo|busco|me gustar[ií]a|saber|entender|comprender|"
    r"anticipar|predecir|deseo|queremos|necesitamos|tenemos|mi objetivo|el objetivo)\b",
    re.IGNORECASE,
)


def looks_like_name(text: str) -> bool:
    """Un responsable es un nombre o rol corto, no una frase sobre el objetivo."""

    clean = text.strip()
    return (
        bool(clean)
        and len(clean) <= ACTOR_MAX_CHARS
        and len(clean.split()) <= _ACTOR_MAX_WORDS
        and not any(mark in clean for mark in "?¿\n")
        and not _GOAL_START.match(clean)
    )


def _render_actor(
    service: ContextDraftService, stored: StoredDraft, suggested: str = ""
) -> str | None:
    item = stored.draft.get(DraftField.DECISION_ACTOR_LABEL)
    if item.status is ConfirmationStatus.CONFIRMED and item.value:
        st.caption(f"Responsable declarado: {item.value}")
        return str(item.value)
    with st.container(border=True):
        st.markdown("**¿Cuál es tu nombre o tu rol?**")
        render_field_card("decision_actor")
        with st.form(f"actor-{stored.draft_id}", border=False):
            label = st.text_input(
                "Tu nombre o rol para registrar las decisiones",
                value=suggested,  # V2-09: el responsable declarado para el proyecto
                key=f"actor-input-{stored.draft_id}",
                label_visibility="collapsed",
                placeholder="Ej.: Ana, jefa de logística",
                max_chars=ACTOR_MAX_CHARS,
            )
            submitted = st.form_submit_button(
                "Continuar", key=f"submit-{stored.draft_id}-actor", type="primary"
            )
    if submitted:
        if not label.strip():
            st.error("Escribe un nombre o rol para continuar.")
        elif not looks_like_name(label):
            st.error(
                "Eso parece una descripción de lo que quieres lograr. Aquí va sólo tu nombre o "
                "tu rol (por ejemplo «Ana, jefa de logística»); lo que quieres lograr se lo "
                "cuentas a la IA en el paso siguiente."
            )
        else:
            stored.draft.set_by_person(
                DraftField.DECISION_ACTOR_LABEL, label.strip(), actor_label=label.strip()
            )
            _commit(service, stored)
    return None


def _render_conversation(
    service: ContextDraftService,
    stored: StoredDraft,
    profile: DatasetProfileV1,
    provider: LLMProvider | None,
) -> None:
    if provider is None:
        # Recorrido 22/10: sin IA activa, la caja invitaba a «Cuéntale a la IA» y sólo repetía
        # las preguntas de abajo.
        st.info(
            "La IA no está activa en este equipo, así que respondes tú las preguntas de abajo. "
            "Cada una explica qué necesita y trae un ejemplo."
        )
        return
    st.markdown("#### Cuéntale a la IA qué necesitas")
    key = f"assistant-{stored.draft_id}"
    refs = column_refs(profile)
    with st.container(border=True):
        render_field_card("assistant_conversation")
        with st.form(f"{key}-compose", border=False):
            typed = st.text_area(
                "Tu situación, con tus palabras",
                key=f"{key}-message",
                placeholder=(
                    "Ej.: tengo el registro de casos de los últimos meses y quiero saber con "
                    "anticipación cuáles terminarán en un problema."
                ),
            )
            reviewed = st.form_submit_button("Revisar antes de enviar", type="primary")
        if reviewed:
            if typed.strip():
                st.session_state[f"{key}-reviewed"] = typed.strip()
            else:
                st.error("Escribe primero tu situación.")
    message = st.session_state.get(f"{key}-reviewed", "")
    if message:
        with st.container(border=True):
            st.markdown("**Esto es lo que se enviaría a la IA**")
            render_field_card("outgoing_preview")
            excluded = st.multiselect(
                "Columnas que prefieres no mostrarle a la IA (opcional)",
                list(refs),
                format_func=lambda ref: refs[ref],
                key=f"{key}-excluded",
                placeholder="Ninguna",
            )
            # Si se ocultan todas, la IA no tendría con qué ayudar: sólo se ofrece seguir sin
            # IA, y la revisión local usa el perfil completo porque nada sale del equipo.
            all_hidden = bool(refs) and len(excluded) == len(refs)
            safe = SafeProfileForLLMV1.from_profile(
                profile, excluded_refs=set() if all_hidden else set(excluded)
            )
            review = review_outgoing(message, safe)
            if all_hidden:
                st.info(
                    "Ocultaste todas las columnas, así que la IA no tendría con qué ayudarte. "
                    "Deja visible al menos una o continúa sin IA."
                )
            st.code(review.message, language=None, wrap_lines=True)
            st.table(
                []
                if all_hidden
                else [
                    {
                        "Columna": column.name,
                        "Tipo": TYPE_LABELS.get(str(column.inferred_type), "otro"),
                    }
                    for column in review.profile.columns
                ]
            )
            if review.findings:
                st.warning(
                    "Encontramos datos que parecen personales o secretos y los ocultamos: "
                    + ", ".join(sorted({f.kind.lower() for f in review.findings}))
                    + ". Revisa tu texto o excluye esas columnas."
                )
            if review.truncated:
                st.warning("Tu mensaje era muy largo; sólo se enviaría la primera parte.")
            st.caption("El proveedor de IA aplica su propia política de conservación de datos.")
            columns = st.columns(2)
            send = columns[0].button(
                "Enviar a la IA",
                key=f"{key}-send",
                disabled=provider is None or all_hidden,
                type="primary",
            )
            offline = columns[1].button("Continuar sin IA", key=f"{key}-offline")
            if provider is None:
                st.caption("La IA no está activada en este equipo; puedes continuar sin IA.")
        send = send and not all_hidden
        if send or offline:
            st.session_state.pop(f"{key}-reviewed", None)
            with st.spinner("Consultando a la IA…" if send else "Preparando las preguntas…"):
                result = ContextAssistant(provider if send else None).next_turn(
                    review, stored.draft, profile
                )
            apply_turn(stored.draft, result.turn, profile)
            st.session_state[f"{key}-last"] = result.model_dump(mode="json")
            _commit(service, stored)

    last = st.session_state.get(f"{key}-last")
    if last:
        if last["mode"] == "LLM":
            st.success(
                "La IA respondió. Revisa y confirma cada propuesta en las preguntas de abajo."
            )
        elif last["mode"] == "DISABLED":
            # Recorrido 22/10: con la IA activa, elegir «Continuar sin IA» decía que la IA «no
            # está activada en este equipo».
            st.info("Seguiste sin IA: responde tú las preguntas de abajo.")
        else:
            st.warning(
                "La IA no pudo ayudar esta vez: " + fallback_reason_text(last["reason"]) + ". "
                "Puedes volver a enviar tu texto o responder tú las preguntas de abajo."
            )
        if last["mode"] == "LLM":
            st.write(with_column_names(last["turn"]["assistant_message"], refs))
        for question in last["turn"]["questions"]:
            prompt = with_column_names(question["prompt"], refs)
            why = with_column_names(question["why_needed"], refs)
            st.markdown(f"- **{prompt}** — {why}")
        with st.expander("Detalle técnico de la respuesta"):
            st.json(last["trace"], expanded=False)


def _render_text_field(
    service: ContextDraftService, stored: StoredDraft, field: DraftField, actor: str
) -> None:
    copy_key, label = FIELD_LABELS[field]
    item = stored.draft.get(field)
    proposed = item.status is ConfirmationStatus.PROPOSED and item.value is not None
    with st.container(border=True):
        st.markdown(f"**{label}**")
        render_field_card(copy_key)
        _status_line(stored, field)
        if proposed:
            st.caption("La IA propuso este texto. Edítalo si quieres y confírmalo.")
        with st.form(f"field-{stored.draft_id}-{field.value}", border=False):
            value = st.text_area(
                label,
                value="" if item.value is None else str(item.value),
                key=f"text-{stored.draft_id}-{field.value}-r{stored.draft.revision}",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button(
                "Confirmar", key=f"submit-{stored.draft_id}-{field.value}"
            )
    if submitted:
        if len(value.strip()) < 3:
            st.error("Escribe una respuesta de al menos tres caracteres.")
        elif proposed and value.strip() == str(item.value).strip():
            # Sin cambios: se confirma la propuesta y se conserva su origen (IA).
            stored.draft.confirm(field, actor_label=actor)
            _commit(service, stored)
        else:
            stored.draft.set_by_person(field, value.strip(), actor_label=actor)
            _commit(service, stored)


def _readable_value(value: Any) -> str:
    if isinstance(value, bool):
        return "verdadero" if value else "falso"
    return str(value)


def _thousands(count: int) -> str:
    return f"{count:,}".replace(",", ".")


def event_option_label(target: str, value: Any, count: int) -> str:
    """UX-03: opción del evento en lenguaje natural, sin comillas de código."""

    return f"Que «{target}» sea «{_readable_value(value)}» ({_thousands(count)} casos)"


def majority_event_warning(value: Any, counts: dict[Any, int]) -> str | None:
    """Advierte cuando el evento elegido es el resultado más frecuente."""

    total = sum(counts.values())
    count = counts.get(value)
    if count is None or total == 0 or count * 2 <= total:
        return None
    return (
        f"Elegiste el resultado más común ({_thousands(count)} de {_thousands(total)} casos). "
        "Normalmente se quiere detectar el caso menos frecuente, como el atraso o la falla. "
        "Confírmalo sólo si de verdad quieres anticipar este resultado."
    )


def _choice(
    service: ContextDraftService,
    stored: StoredDraft,
    field: DraftField,
    actor: str,
    options: list[Any],
    labels: dict[Any, str] | None = None,
    *,
    radio: bool = False,
    warning: Callable[[Any], str | None] | None = None,
    default: Any | None = None,
) -> None:
    with st.container(border=True):
        _choice_body(
            service,
            stored,
            field,
            actor,
            options,
            labels,
            radio=radio,
            warning=warning,
            default=default,
        )


def _choice_body(
    service: ContextDraftService,
    stored: StoredDraft,
    field: DraftField,
    actor: str,
    options: list[Any],
    labels: dict[Any, str] | None,
    *,
    radio: bool,
    warning: Callable[[Any], str | None] | None,
    default: Any | None = None,
) -> None:
    copy_key, label = FIELD_LABELS[field]
    item = stored.draft.get(field)
    st.markdown(f"**{label}**")
    render_field_card(copy_key)
    _status_line(stored, field, has_default=default is not None)
    if item.status is ConfirmationStatus.PROPOSED and item.value in options:
        shown = labels.get(item.value, item.value) if labels else item.value
        st.markdown(f"Propuesta de la IA: *{shown}*")
    _accept_button(service, stored, field, actor)
    confirmed = item.status is ConfirmationStatus.CONFIRMED and item.value in options
    # Un valor por defecto se muestra marcado, pero no se guarda hasta que la persona lo cambie.
    untouched_default = (
        default in options and item.status is ConfirmationStatus.EMPTY and not confirmed
    )
    index = options.index(item.value) if confirmed else None
    if untouched_default:
        index = options.index(default)
    widget = st.radio if radio else st.selectbox
    kwargs: dict[str, Any] = {
        "index": index,
        "key": f"choice-{stored.draft_id}-{field.value}-r{stored.draft.revision}",
        "label_visibility": "collapsed",
    }
    if labels:
        kwargs["format_func"] = lambda value: labels.get(value, str(value))
    if not radio:
        kwargs["placeholder"] = "Elige una opción"
    selected = widget(label, options, **kwargs)
    message = None
    if warning is not None:
        shown_value = selected if selected is not None else (item.value if confirmed else None)
        message = warning(shown_value) if shown_value is not None else None
        if message:
            st.warning(message)
    if selected is None or (confirmed and selected == item.value):
        return
    if untouched_default and selected == default:
        return
    # Elegir una opción ya es la decisión de la persona: se guarda al instante. Sólo cuando
    # hay una advertencia se pide confirmar de nuevo, para que la lea antes de seguir.
    if message is None or st.button(
        "Confirmar de todas formas", key=f"save-{stored.draft_id}-{field.value}"
    ):
        stored.draft.set_by_person(field, selected, actor_label=actor)
        _commit(service, stored)


# UX-18: una sola pregunta por columna; cada respuesta fija uso y disponibilidad.
FEATURE_ANSWERS: dict[str, tuple[DraftUse, Availability, str]] = {
    "use": (DraftUse.INCLUDE, Availability.YES, "Sí, lo conozco: que lo use"),
    "known_skip": (DraftUse.EXCLUDE, Availability.YES, "Sí, pero que no lo use"),
    "after": (DraftUse.EXCLUDE, Availability.NO, "No, se sabe después"),
    "unknown": (DraftUse.UNKNOWN, Availability.UNKNOWN, "No sé (queda fuera)"),
}


def feature_answer(use: DraftUse, available: Availability) -> str | None:
    for key, (answer_use, answer_available, _) in FEATURE_ANSWERS.items():
        if (answer_use, answer_available) == (use, available):
            return key
    if available is Availability.UNKNOWN or use is DraftUse.UNKNOWN:
        return "unknown"
    return "after" if available is Availability.NO else "known_skip"


def unsafe_columns(profile: DatasetProfileV1) -> dict[str, str]:
    """Columnas que `core/` nunca deja usar, con la razón en lenguaje común."""

    reasons: dict[str, str] = {}
    for column in profile.columns:
        if column.constant:
            reasons[column.name] = "tiene el mismo valor en todas las filas"
        elif column.possible_identifier:
            reasons[column.name] = "parece un identificador (distinto en cada fila)"
    return reasons


def bulk_candidates(stored: StoredDraft, unsafe: Collection[str] = ()) -> list[str]:
    """Columnas que la IA revisó, propone usar y no marcó como posteriores al evento."""

    return [
        item.column
        for item in stored.draft.candidate_features()
        if item.status is ConfirmationStatus.PROPOSED
        and item.note_source in {FieldSource.LLM, FieldSource.SAVED_CONTEXT}
        and item.use is DraftUse.INCLUDE
        and not item.possible_post_event
        and item.column not in unsafe
    ]


_GROUP_TITLES = {
    FeatureGroup.USE: "Recomendado usar",
    FeatureGroup.EXCLUDE: "Recomendado dejar fuera",
    FeatureGroup.ASK: "Necesita tu respuesta",
}


def _render_feature_groups(
    service: ContextDraftService,
    stored: StoredDraft,
    actor: str,
    profile: DatasetProfileV1,
    target: str,
) -> None:
    """V2-10: columnas agrupadas con su motivo; aceptar en bloque sólo lo seguro, y deshacer."""

    groups = group_features(stored.draft, profile, target_column=target)
    undo_key = f"feature-undo-{stored.draft_id}"
    with st.container(border=True):
        st.markdown("**Goal Settings revisó tus columnas**")
        for group, title in _GROUP_TITLES.items():
            items = [item for item in groups.items if item.group is group]
            if not items:
                continue
            with st.expander(f"{title} ({len(items)})", expanded=group is FeatureGroup.ASK):
                for item in items:
                    mark = " · confirmado" if item.confirmed else ""
                    st.markdown(f"- **{item.column}**{mark}: {item.reason}")
        pending = [
            item
            for item in groups.items
            if not item.confirmed and item.group is not FeatureGroup.ASK
        ]
        if pending and st.button(
            f"Aceptar recomendaciones ({len(pending)} columnas)",
            key=f"feature-accept-{stored.draft_id}",
            type="primary",
        ):
            accept_recommendations(stored.draft, groups, actor_label=actor)
            st.session_state[undo_key] = [item.column for item in pending]
            _commit(service, stored)
        if st.session_state.get(undo_key) and st.button(
            "Deshacer las recomendaciones aceptadas", key=f"feature-undo-btn-{stored.draft_id}"
        ):
            undo_recommendations(stored.draft, st.session_state.pop(undo_key), actor_label=actor)
            _commit(service, stored)
        st.caption(
            "Las de «Necesita tu respuesta» las contestas tú abajo: sólo tú sabes si ese dato se "
            "conoce antes del resultado."
        )


def _render_features(
    service: ContextDraftService, stored: StoredDraft, actor: str, profile: DatasetProfileV1
) -> None:
    st.markdown("#### ¿Qué datos puede mirar el sistema?")
    render_field_card("feature_availability")
    render_field_card("feature_use")
    unsafe = unsafe_columns(profile)

    target = stored.draft.get(DraftField.TARGET_COLUMN)
    if target.status is ConfirmationStatus.CONFIRMED and target.value:
        _render_feature_groups(service, stored, actor, profile, str(target.value))
    bulk = [] if target.status is ConfirmationStatus.CONFIRMED else bulk_candidates(stored, unsafe)
    if bulk:
        with st.container(border=True):
            st.markdown(
                f"**La IA revisó {len(bulk)} columnas sin alertas:** "
                + ", ".join(f"«{column}»" for column in bulk)
                + "."
            )
            agreed = st.checkbox(
                "Confirmo que conozco todos estos datos antes de decidir y que el sistema "
                "puede usarlos",
                key=f"bulk-check-{stored.draft_id}-r{stored.draft.revision}",
            )
            if st.button(
                f"Confirmar estas {len(bulk)} columnas",
                key=f"bulk-{stored.draft_id}",
                disabled=not agreed,
            ):
                for column in bulk:
                    stored.draft.set_feature(
                        column,
                        use=DraftUse.INCLUDE,
                        available_at_prediction=Availability.YES,
                        actor_label=actor,
                    )
                _commit(service, stored)

    options = list(FEATURE_ANSWERS)
    for item in stored.draft.candidate_features():
        with st.container(border=True):
            st.markdown(
                f"**¿Conoces «{item.column}» antes de decidir?** · {STATUS_LABELS[item.status]}"
            )
            if item.note:
                origin = SOURCE_LABELS.get(item.note_source, "") if item.note_source else ""
                st.caption(f"{origin}: {item.note}" if origin else item.note)
            if item.possible_post_event:
                st.warning(
                    "La IA cree que este dato podría conocerse recién después del resultado. "
                    "Si es así, no sirve para anticiparlo: elige «No, se sabe después»."
                )
            confirmed = item.status is ConfirmationStatus.CONFIRMED
            current = feature_answer(item.use, item.available_at_prediction) if confirmed else None
            allowed = options
            if item.column in unsafe:
                # UX-18: no se ofrece una respuesta que `core/` rechazaría al guardar.
                allowed = [key for key in options if key != "use"]
                st.info(
                    f"Esta columna {unsafe[item.column]}, así que no ayuda a anticipar el "
                    "resultado y el sistema no la usará."
                )
                if current == "use":
                    st.warning("Habías elegido que la use: elige otra respuesta.")
                    current = None
            choice = st.radio(
                f"¿Conoces «{item.column}» antes de decidir?",
                allowed,
                index=allowed.index(current) if current else None,
                format_func=lambda key: FEATURE_ANSWERS[key][2],
                horizontal=True,
                label_visibility="collapsed",
                key=f"feature-{stored.draft_id}-{item.column}-r{stored.draft.revision}",
            )
            if choice is not None and choice != current:
                use, available, _ = FEATURE_ANSWERS[choice]
                stored.draft.set_feature(
                    item.column, use=use, available_at_prediction=available, actor_label=actor
                )
                _commit(service, stored)


def _missing_text(stored: StoredDraft) -> list[str]:
    labels = []
    for name in stored.draft.missing_confirmations():
        if name.startswith("feature:"):
            labels.append(f"Columna «{name.removeprefix('feature:')}»")
        elif name == "features":
            labels.append("Revisión de variables")
        else:
            labels.append(FIELD_LABELS[DraftField(name)][1])
    return labels


def _render_reuse(
    service: ContextDraftService,
    stored: StoredDraft,
    repository: WorkflowRepository,
    actor: str,
) -> None:
    """GA-13: partir de lo ya decidido en otro proyecto; todo llega como propuesta."""

    _render_bundle_import(service, stored, actor)
    saved = repository.list_reusable_contexts(stored.draft.project_id)
    if not saved:
        return
    labels = {
        str(row["id"]): f"{row['project_name']} · resultado «{row['target_column']}»"
        for row in saved
    }
    with st.expander("Partir de un contexto guardado"):
        st.caption(
            "Trae como propuestas lo que se decidió en otro proyecto: objetivo, costos de cada "
            "error, medida principal y qué columnas usar (sólo las que existen en este archivo). "
            "Igual confirmas o cambias cada respuesta."
        )
        with st.form(f"reuse-context-{stored.draft_id}", clear_on_submit=True):
            chosen = st.selectbox(
                "Contexto guardado", list(labels), format_func=lambda value: labels[value]
            )
            submitted = st.form_submit_button("Usar como propuesta")
        if submitted:
            try:
                proposed = service.reuse(stored, chosen, actor_label=actor)
            except (LookupError, ValueError, ContextValidationError) as error:
                render_error(error, fallback_code="INVALID_CONTEXT")
                return
            st.success(f"Se propusieron {proposed} respuestas. Revísalas y confírmalas abajo.")


def _render_bundle_import(service: ContextDraftService, stored: StoredDraft, actor: str) -> None:
    """C2-04: decisiones de una receta exportada de otro proyecto, como propuestas."""

    with st.expander("Reusar las decisiones de otro proyecto (opcional)"):
        st.caption(
            "Si en otro proyecto guardaste sus decisiones con «Guardar decisiones para otro "
            "proyecto» (en «Usar el modelo»), sube aquí ese archivo (termina en .json). Llegan "
            "como propuestas que igual confirmas; ningún dato de aquel proyecto viaja en él."
        )
        with st.form(f"bundle-import-{stored.draft_id}", clear_on_submit=True):
            upload = st.file_uploader("Archivo de decisiones (.json)", type=["json"])
            submitted = st.form_submit_button("Importar como propuesta")
        if not submitted:
            return
        if upload is None:
            st.error("Selecciona el archivo de decisiones.")
            return
        try:
            proposed = service.reuse_from_bundle(
                stored, upload.getvalue(), filename=upload.name, actor_label=actor
            )
        except (RecipeBundleError, ValueError) as error:
            render_error(error, fallback_code="INVALID_CONTEXT")
            return
        st.success(f"Se propusieron {proposed} respuestas. Revísalas y confírmalas abajo.")


def render_context_assistant(
    repository: WorkflowRepository,
    storage_root: Path,
    project_id: str,
    dataset: Any,
    provider: LLMProvider | None,
) -> None:
    service = ContextDraftService(repository, storage_root)
    try:
        stored = service.open(project_id, dataset["id"])
    except ContextValidationError as error:
        render_error(error, fallback_code="INVALID_CONTEXT")
        return
    profile = DatasetProfileV1.model_validate(json.loads(dataset["profile_json"]))

    roles = repository.get_project_roles(project_id)
    actor = _render_actor(service, stored, "" if roles is None else str(roles["owner_label"]))
    if actor is None:
        return
    _render_reuse(service, stored, repository, actor)
    _render_conversation(service, stored, profile, provider)

    st.markdown("#### Tus respuestas")
    st.caption("Confirma cada respuesta por separado: la IA sólo propone.")
    _render_text_field(service, stored, DraftField.OBJECTIVE, actor)
    _choice(
        service,
        stored,
        DraftField.TASK_TYPE,
        actor,
        list(TASK_TYPES),
        TASK_TYPES,
        radio=True,
        default="binary_classification",
    )
    task_type = stored.draft.task_type
    candidates = target_candidates(profile, task_type)
    _choice(service, stored, DraftField.TARGET_COLUMN, actor, candidates)
    target = stored.draft.get(DraftField.TARGET_COLUMN)
    if target.status is ConfirmationStatus.CONFIRMED:
        try:
            support = BusinessContextService(repository, storage_root).inspect_target(
                dataset["id"], str(target.value)
            )
        except ContextValidationError as error:
            render_error(error, fallback_code="INVALID_CONTEXT")
            support = []
        if task_type == "binary_classification" and len(support) == 2:
            # El menos frecuente primero; ninguno viene elegido.
            ordered = sorted(support, key=lambda entry: entry.count)
            values = [entry.value for entry in ordered]
            counts = {entry.value: entry.count for entry in ordered}
            _choice(
                service,
                stored,
                DraftField.POSITIVE_CLASS,
                actor,
                values,
                {
                    entry.value: event_option_label(str(target.value), entry.value, entry.count)
                    for entry in ordered
                },
                radio=True,
                warning=lambda value: majority_event_warning(value, counts),
            )
    else:
        st.caption("Elige primero el resultado para ver sus dos valores posibles.")
    for field in TEXT_FIELDS[1:4]:
        _render_text_field(service, stored, field, actor)
    metrics = METRICS_BY_TASK_TYPE[task_type]
    _choice(
        service,
        stored,
        DraftField.PRIMARY_METRIC,
        actor,
        metrics,
        {key: METRIC_QUESTION.get(key, METRIC_LABELS[key]) for key in metrics},
        radio=True,
    )
    _render_text_field(service, stored, DraftField.METRIC_RATIONALE, actor)
    _choice(
        service,
        stored,
        DraftField.GUARDRAIL_MODE,
        actor,
        # La condición mínima usa métricas de sí/no: en otros tipos no se ofrece.
        list(GUARDRAIL_MODES)
        if task_type == "binary_classification"
        else ["no_quantitative_guardrail"],
        GUARDRAIL_MODES,
        radio=True,
    )
    if stored.draft.get(DraftField.GUARDRAIL_MODE).value == "quantitative":
        _choice(
            service,
            stored,
            DraftField.GUARDRAIL_METRIC,
            actor,
            metrics,
            {key: METRIC_LABELS[key] for key in metrics},
        )
        threshold = stored.draft.get(DraftField.GUARDRAIL_THRESHOLD)
        render_field_card("guardrail_threshold", state=STATUS_LABELS[threshold.status])
        with st.form(f"threshold-{stored.draft_id}"):
            value = st.number_input(
                "Valor mínimo (entre 0 y 1)",
                min_value=0.0,
                max_value=1.0,
                value=None if threshold.value is None else float(threshold.value),
                step=0.05,
            )
            submitted = st.form_submit_button(
                "Guardar valor mínimo", key=f"submit-{stored.draft_id}-threshold"
            )
        if submitted:
            if value is None:
                st.error("Escribe un valor entre 0 y 1.")
            else:
                stored.draft.set_by_person(
                    DraftField.GUARDRAIL_THRESHOLD, float(value), actor_label=actor
                )
                _commit(service, stored)
        _render_text_field(service, stored, DraftField.GUARDRAIL_RATIONALE, actor)

    _render_features(service, stored, actor, profile)

    st.markdown("#### Guardar tus decisiones")
    missing = _missing_text(stored)
    if missing:
        shown = missing[:6]
        extra = f"\n- … y {len(missing) - len(shown)} más" if len(missing) > len(shown) else ""
        heading = (
            "Te falta confirmar 1 respuesta para poder guardar:"
            if len(missing) == 1
            else f"Te faltan {len(missing)} respuestas por confirmar para poder guardar:"
        )
        st.info(heading + "\n" + "\n".join(f"- {item}" for item in shown) + extra)
    else:
        st.success("Todo confirmado. Ya puedes guardar tus decisiones.")
    columns = st.columns(2)
    if columns[0].button(
        "Guardar mis decisiones",
        key=f"finalize-{stored.draft_id}",
        disabled=bool(missing),
        type="primary",
    ):
        try:
            result = service.finalize(stored)
        except DraftNotReadyError:
            st.error("Todavía faltan decisiones por confirmar.")
        except (ContextValidationError, InvalidStateTransitionError, ValueError) as error:
            render_error(error, fallback_code=getattr(error, "code", str(error)))
        else:
            st.session_state.pop(f"assistant-{stored.draft_id}-last", None)
            st.success(f"Decisiones guardadas (versión {result.version_no}).")
            st.rerun()
    confirm_discard = columns[1].checkbox(
        "Quiero descartar este borrador", key=f"discard-check-{stored.draft_id}"
    )
    if confirm_discard and columns[1].button(
        "Descartar borrador", key=f"discard-{stored.draft_id}"
    ):
        service.discard(stored)
        st.session_state.pop(f"assistant-{stored.draft_id}-last", None)
        st.rerun()

"""MV-05: decisiones y escenarios revisables, incluso antes de disponer de datos."""

from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4

import streamlit as st

from core.db.repositories import WorkflowRepository
from core.domain.capabilities import CapabilityService
from core.domain.decision_value import DecisionValueService
from core.domain.errors import DomainError
from core.domain.value_policy import ValuePolicyService
from core.economics.scenarios import evaluate_scenario_version


def _optional(value: str) -> str | None:
    return value.strip() or None


def _number(value: str, *, integer: bool = False) -> Decimal | int | None:
    if not value.strip():
        return None
    try:
        return int(value) if integer else Decimal(value.strip().replace(",", "."))
    except (ValueError, InvalidOperation) as error:
        raise ValueError("Escribe números sin separadores de miles.") from error


def _operation(key: str) -> str:
    return st.session_state.setdefault(f"value-operation-{key}", str(uuid4()))


def _saved(key: str) -> None:
    st.session_state.pop(f"value-operation-{key}", None)
    st.rerun()


def _decision(service: DecisionValueService, capability_id: str) -> None:
    versions = service.list_decision_versions(capability_id)
    latest = versions[-1] if versions else None
    values = latest.spec.model_dump() if latest else {}
    key = f"decision-value-{capability_id}-{latest.id if latest else 'new'}"
    predictive = {
        item.id: item
        for item in service.list_predictive_versions(capability_id)
        if item.confirmed_human
    }
    economic = {
        item.id: item
        for item in service.list_economic_versions(capability_id)
        if item.confirmed_human
    }
    if latest:
        status = "confirmada" if latest.confirmed_human else "borrador"
        st.caption(f"Decisión guardada · versión {latest.version_no} · {status}")
        labels = {
            "question": "qué decisión quieres mejorar",
            "action": "qué acción podrías tomar",
            "alternative": "qué harías sin esa acción",
            "kpi": "cómo medirás el resultado",
            "owner_label": "quién es responsable de la decisión",
        }
        missing = [labels[field] for field in latest.missing_fields if field in labels]
        if missing:
            st.info("Tu borrador está guardado. Falta definir: " + "; ".join(missing) + ".")
        else:
            st.success("La decisión de negocio está descrita y guardada.")
        for field, label in (
            ("question", "Decisión"),
            ("action", "Acción propuesta"),
            ("kpi", "Indicador de negocio"),
        ):
            if values.get(field):
                st.write(f"**{label}:** {values[field]}")
        if any(field.startswith("predictive_") for field in latest.missing_fields):
            st.caption(
                "Ya hay una definición predictiva confirmada. Abre Editar decisión y "
                "selecciónala en Restricciones y evidencia vinculada."
                if predictive
                else "Para vincular una predicción, continúa en Evaluaciones y actualizaciones: "
                "carga datos y define qué resultado quieres anticipar."
            )
        if any(field.startswith("economic_") for field in latest.missing_fields):
            st.caption(
                "Ya hay un escenario económico confirmado. Abre Editar decisión y "
                "selecciónalo en Restricciones y evidencia vinculada."
                if economic
                else "Para estimar valor, abre Estimar ingresos o ahorro. Después podrás "
                "vincular el escenario confirmado a esta decisión."
            )
    else:
        st.caption(
            "Describe la decisión y cómo sabrás si sirve. Puedes guardar un borrador "
            "con lo que ya sabes y completarlo después."
        )
    with (
        st.expander("Editar decisión" if latest else "Completar decisión", expanded=latest is None),
        st.form(key),
    ):
        fields = {}
        for name, label, example in (
            ("question", "Qué decisión quieres mejorar", "A quién ofrecer ayuda para comprar"),
            ("action", "Qué acción podrías tomar", "Contactar a quienes podrían abandonar"),
            ("alternative", "Alternativa sin intervención", "Mantener la atención habitual"),
            ("kpi", "Indicador de negocio", "Margen de primeras compras o costo de atención"),
            ("owner_label", "Responsable declarado", "Persona o equipo que tomará la decisión"),
        ):
            fields[name] = _optional(
                st.text_input(label, value=values.get(name) or "", placeholder=example)
            )
        with st.expander("Restricciones y evidencia vinculada"):
            constraints = st.text_area(
                "Restricciones (una por línea)", value="\n".join(values.get("constraints", []))
            )
            for field, label, choices in (
                ("predictive_version_id", "Definición predictiva de esta decisión", predictive),
                ("economic_version_id", "Escenario económico de esta decisión", economic),
            ):
                options = [None, *choices]
                previous = values.get(field)
                if not choices:
                    fields[field] = None
                    st.caption(f"{label}: podrás elegirla cuando exista una versión confirmada.")
                    continue
                fields[field] = st.selectbox(
                    label,
                    options,
                    index=options.index(previous) if previous in options else 0,
                    format_func=lambda value, items=choices: (
                        "Pendiente" if value is None else f"Versión {items[value].version_no}"
                    ),
                )
        actor = st.text_input("Quién registra la decisión")
        confirmed = st.checkbox("Confirmo esta decisión de negocio", value=False)
        submitted = st.form_submit_button("Guardar decisión")
    if submitted:
        fields["constraints"] = [line.strip() for line in constraints.splitlines() if line.strip()]
        try:
            service.create_decision_version(
                capability_id,
                spec=fields,
                based_on_version_id=latest.id if latest else None,
                confirmed_human=confirmed,
                actor_label=actor,
                operation_id=_operation(key),
            )
        except (ValueError, DomainError) as error:
            st.error(str(error))
        else:
            _saved(key)


def _economy(service: DecisionValueService, capability_id: str, *, value_kind: str) -> None:
    versions = service.list_economic_versions(capability_id)
    latest = versions[-1] if versions else None
    values = latest.scenario.model_dump() if latest else {}
    key = f"economic-value-{capability_id}-{latest.id if latest else 'new'}"
    st.markdown("#### Escenario económico")
    st.caption(
        "Exploración con supuestos. La tasa corresponde al grupo que se atendería; el modelo "
        "ordena casos y no demuestra cuánto cambia el resultado al intervenir. Estos escenarios "
        "son de demostración sintética, no dinero conseguido por un negocio."
    )
    revenue = value_kind == "revenue"
    st.write(
        "Aquí los ingresos se estiman como margen recuperado al evitar una pérdida: "
        "por ejemplo, ayudar a completar una compra que se habría abandonado."
        if revenue
        else "Aquí el ahorro se estima como costo evitado al reducir un problema: "
        "por ejemplo, evitar un retraso que generaría trabajo adicional."
    )
    labels = (
        ("currency", "Moneda (código de 3 letras)"),
        ("currency_minor_units", "Decimales de la moneda"),
        ("period", "Período del escenario"),
        ("planned_cases", "Casos del período"),
        ("capacity_cases", "Máximo de casos que puedes atender (opcional)"),
        (
            "external_risk_rate",
            "Proporción con riesgo de perder ingresos"
            if revenue
            else "Proporción con el problema sin la acción",
        ),
        ("risk_rate_basis", "Fuente o fundamento de la tasa"),
        (
            "relative_effect",
            "Reducción relativa supuesta de esa pérdida"
            if revenue
            else "Reducción relativa supuesta del problema",
        ),
        ("relative_effect_basis", "Fundamento de la reducción"),
        (
            "value_per_case",
            "Margen recuperable por caso (supuesto)"
            if revenue
            else "Costo evitado por problema resuelto (supuesto)",
        ),
        ("action_cost_per_case", "Costo por intervención"),
        ("fixed_cost", "Costo fijo al intervenir"),
        ("value_basis", "Fundamento del valor económico"),
    )
    integers = {"currency_minor_units", "planned_cases", "capacity_cases"}
    decimals = {
        "external_risk_rate",
        "relative_effect",
        "value_per_case",
        "action_cost_per_case",
        "fixed_cost",
    }
    examples = {
        "currency": "CLP, USD…",
        "currency_minor_units": "0 para CLP; 2 para USD",
        "period": "Por ejemplo: una semana de pedidos",
        "planned_cases": "Cantidad total de casos del período",
        "capacity_cases": "Vacío si no hay un límite adicional",
        "external_risk_rate": "0,20 equivale al 20 % del grupo que atenderías",
        "relative_effect": "0,25 significa reducir el problema un 25 %; no 25 puntos",
        "risk_rate_basis": "De dónde viene esta estimación",
        "relative_effect_basis": "Por qué esperas ese cambio al actuar",
        "value_basis": "Cómo estimaste el margen o costo por caso",
    }
    with (
        st.expander(
            "Editar supuestos" if latest else "Completar supuestos", expanded=latest is None
        ),
        st.form(key),
    ):
        raw = {}
        for index, (name, label) in enumerate(labels):
            if index % 2 == 0:
                columns = st.columns(2)
            value = values.get(name)
            raw[name] = columns[index % 2].text_input(
                label,
                value="" if value is None else str(value),
                help=examples.get(name),
            )
        st.caption(
            "Tasas entre 0 y 1. La capacidad es opcional. Los campos vacíos se guardan como "
            "faltantes; no se sustituyen por ceros."
        )
        actor = st.text_input("Quién registra el escenario")
        confirmed = st.checkbox("Confirmo estos supuestos para explorar el escenario", value=False)
        submitted = st.form_submit_button("Guardar escenario")
    if submitted:
        try:
            scenario = {
                name: (
                    _number(value, integer=name in integers)
                    if name in integers | decimals
                    else _optional(value)
                )
                for name, value in raw.items()
            }
            service.create_economic_version(
                capability_id,
                scenario=scenario,
                based_on_version_id=latest.id if latest else None,
                confirmed_human=confirmed,
                actor_label=actor,
                operation_id=_operation(key),
            )
        except (ValueError, DomainError) as error:
            st.error(str(error))
        else:
            _saved(key)
    if latest:
        st.caption(
            f"Escenario versión {latest.version_no} · "
            f"{'supuestos confirmados' if latest.confirmed_human else 'borrador'}"
        )
        if latest.missing_fields:
            label_by_field = dict(labels)
            st.info(
                "Escenario guardado. Para calcular falta: "
                + "; ".join(label_by_field.get(field, field) for field in latest.missing_fields)
                + "."
            )
        if not latest.confirmed_human:
            st.info("Completa y confirma los supuestos para calcular el escenario.")
            return
        result = evaluate_scenario_version(latest)
        if result.status != "scenario_estimate":
            st.info("Falta confirmar información para calcular este escenario.")
            return
        currency = latest.scenario.currency
        columns = st.columns(3)
        columns[0].metric("Exposición del grupo", f"{currency} {result.exposure_value}")
        columns[1].metric("Beneficio bruto supuesto", f"{currency} {result.gross_value}")
        columns[2].metric("Valor neto supuesto", f"{currency} {result.net_value}")
        st.caption(
            f"{result.selected_cases} intervenciones supuestas. Costos variables: "
            f"{currency} {result.action_cost}; costo fijo: {currency} {result.fixed_cost_applied}. "
            "La exposición es valor en riesgo; no equivale a valor recuperable. "
            "El beneficio depende de la efectividad que declaraste y todavía debe medirse."
        )


def _prediction(
    repository: WorkflowRepository, service: DecisionValueService, capability_id: str
) -> None:
    cycles = CapabilityService(repository).list_cycles(capability_id)
    contexts = {}
    for cycle in cycles:
        context = repository.get_latest_context(cycle.project_id)
        if context is not None:
            contexts[cycle.id] = (cycle, context)
    with st.expander("Qué debería predecirse"):
        if not contexts:
            st.info(
                "Podrás definir qué predecir cuando una evaluación tenga un resultado y "
                "datos definidos. El objetivo de negocio ya puede quedar guardado."
            )
            return
        cycle_id = st.selectbox(
            "Evaluación con definición del resultado",
            list(contexts),
            format_func=lambda key: (
                f"Evaluación {contexts[key][0].cycle_no}: {contexts[key][0].reason}"
            ),
            key=f"predictive-source-{capability_id}",
        )
        context = contexts[cycle_id][1]
        versions = service.list_predictive_versions(capability_id)
        latest = versions[-1] if versions else None
        key = f"predictive-value-{capability_id}-{context['id']}-{latest.id if latest else 'new'}"
        try:
            draft = service.draft_predictive_from_context(
                capability_id, context_version_id=str(context["id"])
            )
        except (ValueError, DomainError) as error:
            st.info(str(error))
            return
        st.write(f"Resultado: {draft.target_column}. Momento: {draft.prediction_moment}.")
        if draft.snapshot_recipe is not None:
            st.caption(
                "Esta definición conserva la receta registrada de la tabla. "
                "Revisa que el horizonte declarado coincida con el resultado de esa receta."
            )
        with st.form(key):
            population = st.text_input("Población a la que se aplica")
            definition = st.text_area("Qué significa observar ese resultado")
            window_kind = st.selectbox(
                "Cómo termina la observación",
                ["fixed_days", "deadline_column"],
                format_func=lambda value: (
                    "Días desde la decisión" if value == "fixed_days" else "Fecha límite por caso"
                ),
            )
            days = st.text_input("Días de observación")
            deadline = st.text_input("Columna de fecha límite")
            actor = st.text_input("Quién registra la definición predictiva")
            confirmed = st.checkbox("Confirmo esta definición predictiva", value=False)
            submitted = st.form_submit_button("Guardar definición predictiva")
        if submitted:
            try:
                payload = draft.model_dump(mode="json")
                payload["population"] = _optional(population)
                payload["target_definition"] = _optional(definition)
                payload["outcome_window"] = (
                    {"kind": "fixed_days", "days": _number(days, integer=True)}
                    if window_kind == "fixed_days" and days.strip()
                    else (
                        {"kind": "deadline_column", "column": deadline.strip()}
                        if window_kind == "deadline_column" and deadline.strip()
                        else None
                    )
                )
                service.create_predictive_version(
                    capability_id,
                    spec=payload,
                    based_on_version_id=latest.id if latest else None,
                    confirmed_human=confirmed,
                    actor_label=actor,
                    operation_id=_operation(key),
                )
            except (ValueError, DomainError) as error:
                st.error(str(error))
            else:
                _saved(key)
        binding = service.get_cycle_binding(cycle_id)
        if binding is not None:
            st.caption("Esta evaluación ya conserva su definición predictiva vinculada.")
        elif latest is not None and latest.confirmed_human:
            with st.form(f"bind-predictive-{cycle_id}-{latest.id}"):
                st.write(f"Vincular esta evaluación a la definición {latest.version_no}.")
                actor = st.text_input("Quién vincula la definición a la evaluación")
                submitted = st.form_submit_button("Vincular definición a la evaluación")
            if submitted:
                key = f"bind-predictive-{cycle_id}-{latest.id}"
                try:
                    service.bind_predictive_to_cycle(
                        capability_id,
                        cycle_id=cycle_id,
                        predictive_version_id=latest.id,
                        actor_label=actor,
                        operation_id=_operation(key),
                    )
                except (ValueError, DomainError) as error:
                    st.error(str(error))
                else:
                    _saved(key)


def render_capability_value(
    repository: WorkflowRepository, capability_id: str, *, storage_root: str | Path | None = None
) -> None:
    service = DecisionValueService(repository)
    with st.expander("Decisión de negocio", expanded=True):
        _decision(service, capability_id)
    with st.expander("Estimar ingresos o ahorro"):
        _economy(
            service,
            capability_id,
            value_kind=CapabilityService(repository).get_capability(capability_id).value_kind,
        )
    _prediction(repository, service, capability_id)
    with st.expander("A quién atender y cómo revisar la decisión"):
        _policy(repository, capability_id)
    if storage_root is not None:
        with st.expander("Comparar una actualización del modelo"):
            from app.ui.promotion_view import render_promotion

            render_promotion(repository, storage_root, capability_id)


def _policy(repository: WorkflowRepository, capability_id: str) -> None:
    st.markdown("**Regla para priorizar la atención**")
    st.caption(
        "Esta regla propone a quién atender según el modelo, tu capacidad y los costos "
        "declarados. Revisarla es una decisión separada de aprobar el modelo."
    )
    service = ValuePolicyService(repository)
    versions = service.list_versions(capability_id)
    latest = versions[-1] if versions else None
    if latest is not None:
        content = latest.content
        st.caption(
            f"Regla versión {latest.version_no} · seleccionada con datos de comparación · "
            "cálculo económico con supuestos declarados"
        )
        st.metric("Casos propuestos para atención por período", content.plan.to_attend)
        st.metric(
            "Valor neto del escenario, no observado",
            str(content.economic_evaluation.net_value),
        )
        st.caption(
            "Evidencia exploratoria de validación. Esta versión no hereda aprobación "
            "ni una prueba final de políticas anteriores."
        )
        review = service.latest_review(latest.id)
        current = service.is_review_current(latest.id)
        st.write(
            "Revisión de la política: "
            + ("Aprobada y vigente" if current else "Pendiente o desactualizada")
        )
        if review is not None:
            st.caption(
                "Última decisión registrada: "
                + ("Aprobada" if review.decision == "APPROVED" else "Rechazada")
            )
        with st.form(f"policy-review-{latest.id}"):
            with st.expander("Identificador de la versión que estás revisando"):
                st.code(latest.sha256, language=None)
            decision = st.radio(
                "Decisión sobre esta política",
                ["APPROVED", "REJECTED"],
                format_func=lambda value: "Aprobar" if value == "APPROVED" else "Rechazar",
            )
            actor = st.text_input("Quién revisa la política")
            submitted = st.form_submit_button("Registrar revisión de política")
        if submitted:
            key = f"policy-review-{latest.id}"
            try:
                service.review(
                    latest.id,
                    approved_hash=latest.sha256,
                    decision=decision,
                    actor_label=actor,
                    operation_id=_operation(key),
                )
            except (DomainError, ValueError) as error:
                st.error(str(error))
            else:
                _saved(key)

    cycles = CapabilityService(repository).list_cycles(capability_id)
    runs = {}
    for cycle in cycles:
        rows = repository.connection.execute(
            "SELECT r.id FROM runs r JOIN strategy_versions s ON s.id=r.strategy_version_id "
            "JOIN context_versions c ON c.id=s.context_version_id "
            "WHERE c.project_id=? AND r.status='COMPLETED' AND r.selected_model IS NOT NULL "
            "ORDER BY r.started_at DESC",
            (cycle.project_id,),
        ).fetchall()
        for row in rows:
            runs[str(row["id"])] = cycle
    scenarios = [
        item
        for item in DecisionValueService(repository).list_economic_versions(capability_id)
        if item.confirmed_human and not item.missing_fields
    ]
    if not runs or not scenarios:
        st.info(
            "Para proponer esta regla necesitas un modelo recomendado y un escenario "
            "económico confirmado. Continúa con tus datos o completa los supuestos del escenario."
        )
        return
    with st.form(f"policy-proposal-{capability_id}-{latest.id if latest else 'new'}"):
        run_id = st.selectbox(
            "Modelo para la política",
            list(runs),
            format_func=lambda key: f"Evaluación {runs[key].cycle_no} · modelo {key[:8]}",
        )
        scenario_id = st.selectbox(
            "Escenario económico",
            [item.id for item in scenarios],
            format_func=lambda key: (
                "Versión " + str(next(item.version_no for item in scenarios if item.id == key))
            ),
        )
        actor = st.text_input("Quién propone la política")
        submitted = st.form_submit_button("Preparar nueva política")
    if submitted:
        key = f"policy-proposal-{capability_id}-{latest.id if latest else 'new'}"
        try:
            service.propose(
                capability_id,
                cycle_id=runs[run_id].id,
                run_id=run_id,
                economic_version_id=scenario_id,
                based_on_version_id=latest.id if latest else None,
                actor_label=actor,
                operation_id=_operation(key),
            )
        except (DomainError, ValueError) as error:
            st.error(str(error))
        else:
            _saved(key)

"""MV-11: lote vinculado a política y seguimiento explícito de la intervención."""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st

from app.ui.messages import render_error
from core.domain.canonical import canonical_hash
from core.domain.capabilities import CapabilityService
from core.domain.prediction_batches import PredictionBatchError, PredictionBatchService
from core.domain.value_policy import ValuePolicyService


def _capability(repository: Any, run: Any):
    cycle = CapabilityService(repository).get_cycle_for_project(
        repository.get_run_project_id(run["id"])
    )
    return cycle


def _csv_text(value: str) -> str:
    return "'" + value if value.startswith(("=", "+", "-", "@", "\t", "\r")) else value


def render_prediction_batch_offer(
    repository: Any, storage_root: Path, run: Any, scored: Any
) -> None:
    cycle = _capability(repository, run)
    if cycle is None:
        return
    policies = ValuePolicyService(repository)
    versions = policies.list_versions(cycle.capability_id)
    if not versions:
        return
    policy = versions[-1]
    st.markdown("#### Registrar lote con la política revisada")
    if policy.content.run_id != run["id"] or not policies.is_review_current(policy.id):
        st.info("Revisa la política vigente antes de asignar casos de este modelo.")
        return
    events = repository.list_scorings(run["id"])
    event = next(
        (
            row
            for row in events
            if json.loads(row["detail_json"]).get("result_sha256") == canonical_hash(scored)
        ),
        None,
    )
    if event is None:
        st.info("Este resultado no tiene un uso auditado enlazable.")
        return
    with st.form(f"prediction-batch-{run['id']}"):
        period = st.text_input(
            "Período del lote (AAAA-MM)", value=datetime.now(UTC).strftime("%Y-%m")
        )
        classification = st.selectbox(
            "Tipo de datos autorizado",
            ["synthetic", "sandbox", "anonymized"],
            format_func={
                "synthetic": "Sintéticos",
                "sandbox": "Sandbox",
                "anonymized": "Anonimizados",
            }.get,
        )
        basis = st.text_input("Procedencia y autorización de estos casos")
        actor = st.text_input("Responsable del registro del lote")
        submitted = st.form_submit_button("Registrar lote y asignación")
    if not submitted:
        return
    try:
        batch = PredictionBatchService(repository, storage_root).register(
            cycle.capability_id,
            policy_version_id=policy.id,
            scoring=scored,
            scoring_event_id=str(event["id"]),
            period=period,
            data_classification=classification,
            provenance_basis=basis,
            actor_label=actor,
            operation_id=str(uuid4()),
        )
    except (PredictionBatchError, ValueError) as error:
        render_error(error, fallback_code="PREDICTION_BATCH_FAILED")
        return
    st.success(
        f"Lote registrado: {sum(case.assigned for case in batch.cases)} asignados de "
        f"{len(batch.cases)}. Registra aparte quién recibió la intervención."
    )


def render_prediction_batch_tracking(repository: Any, storage_root: Path, run: Any) -> None:
    cycle = _capability(repository, run)
    if cycle is None:
        return
    service = PredictionBatchService(repository, storage_root)
    batches = [
        batch for batch in service.list_batches(cycle.capability_id) if batch.run_id == run["id"]
    ]
    if not batches:
        return
    st.markdown("#### Asignación, actuación y resultado")
    st.caption(
        "Asignar no prueba que se actuó. Un resultado pendiente no cuenta como negativo; "
        "estos conteos no demuestran impacto causal ni dinero ganado o ahorrado."
    )
    labels = {batch.id: f"{batch.period} · {batch.id[:8]}" for batch in batches}
    selected_id = st.selectbox("Lote registrado", list(labels), format_func=labels.get)
    batch = service.get(selected_id)
    summary = service.summary(selected_id)
    st.write(
        f"{summary.cases} casos · {summary.assigned} asignados · {summary.acted} actuados · "
        f"{summary.observed} resultados observados · {summary.pending} pendientes"
    )
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["case_id", "assigned", "score"])
    writer.writerows(
        (_csv_text(case.case_id), "sí" if case.assigned else "no", case.score)
        for case in batch.cases
    )
    st.download_button(
        "Descargar asignación privada",
        data=output.getvalue().encode("utf-8"),
        file_name=f"asignacion-{batch.period}.csv",
        mime="text/csv",
        key=f"prediction-batch-download-{batch.id}",
    )
    with st.form(f"prediction-observation-{selected_id}"):
        case_id = st.selectbox(
            "Caso",
            [case.case_id for case in batch.cases],
        )
        event_type = st.selectbox(
            "Hecho confirmado",
            ["ACTED", "OUTCOME_PENDING", "OUTCOME_OBSERVED"],
            format_func={
                "ACTED": "Intervención efectuada",
                "OUTCOME_PENDING": "Resultado todavía pendiente",
                "OUTCOME_OBSERVED": "Resultado ya observado",
            }.get,
        )
        outcome_positive = st.selectbox(
            "Resultado positivo para el target (sólo si ya se observó)",
            [None, True, False],
            format_func={None: "No corresponde", True: "Sí", False: "No"}.get,
        )
        actor = st.text_input("Quién confirma este hecho")
        submitted = st.form_submit_button("Registrar hecho")
    if not submitted:
        return
    try:
        service.record(
            selected_id,
            case_id=case_id,
            event_type=event_type,
            outcome_positive=outcome_positive,
            occurred_at=datetime.now(UTC),
            actor_label=actor,
            operation_id=str(uuid4()),
        )
    except (PredictionBatchError, ValueError) as error:
        render_error(error, fallback_code="ACTION_OBSERVATION_FAILED")
        return
    st.success("Hecho registrado; los conteos se actualizarán al recargar.")

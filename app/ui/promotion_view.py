"""MV-13: sellar, comparar, revisar y promover sin usar la candidata antes de tiempo."""

from __future__ import annotations

from datetime import UTC, datetime, time
from pathlib import Path

import streamlit as st

from core.db.repositories.workflow import WorkflowRepository
from core.domain.canonical import canonical_hash
from core.domain.promotion import PromotionError, PromotionService
from core.domain.value_policy import ValuePolicyService


def render_promotion(
    repository: WorkflowRepository, storage_root: str | Path, capability_id: str
) -> None:
    service = PromotionService(repository, storage_root)
    versions = ValuePolicyService(repository).list_versions(capability_id)
    current_options = [
        version for version in versions if service.policies.is_review_current(version.id)
    ]
    if len(current_options) < 2 and not service.list_cohorts(capability_id):
        return
    st.markdown("**Comparar y decidir si promover un modelo candidato**")
    active = service.active_policy_id(capability_id)
    if active is not None:
        st.caption(f"Política vigente: {active[:8]}")
    st.caption(
        "La cohorte debe contener resultados maduros de casos sintéticos/sandbox/anonimizados "
        "que ninguno de los modelos ni lotes de intervención usó antes. El criterio se fija "
        "antes de abrirla; el valor monetario mostrado es un escenario, no impacto observado."
    )
    if len(current_options) >= 2:
        labels = {
            version.id: f"v{version.version_no} · run {version.content.run_id[:8]}"
            for version in current_options
        }
        with st.form(f"seal-promotion-{capability_id}"):
            current_id = st.selectbox(
                "Política vigente a comparar", list(labels), format_func=labels.get
            )
            candidate_id = st.selectbox("Política candidata", list(labels), format_func=labels.get)
            file = st.file_uploader("Cohorte común madura (CSV)", type=["csv"])
            classification = st.selectbox(
                "Tipo de datos autorizado", ["synthetic", "sandbox", "anonymized"]
            )
            provenance = st.text_input("Procedencia y autorización de la cohorte")
            coverage_complete = st.checkbox("Confirmo cobertura completa del resultado")
            coverage_basis = st.text_input("Cómo se verificó la cobertura del resultado")
            observed_date = st.date_input("Resultados observados hasta")
            min_rows = st.number_input("Mínimo de casos para decidir", min_value=1, value=30)
            min_delta = st.number_input(
                "Mejora mínima de F1", min_value=0.0, max_value=1.0, value=0.01, step=0.01
            )
            actor = st.text_input("Quién declara el criterio y sella la cohorte")
            seal = st.form_submit_button("Sellar cohorte y criterio")
        if seal:
            if file is None:
                st.error("Sube la cohorte común.")
            else:
                try:
                    service.seal(
                        capability_id,
                        current_policy_id=current_id,
                        candidate_policy_id=candidate_id,
                        filename=file.name,
                        content=file.getvalue(),
                        data_classification=classification,
                        provenance_basis=provenance,
                        coverage_complete=coverage_complete,
                        coverage_basis=coverage_basis,
                        observed_until=datetime.combine(observed_date, time.max, tzinfo=UTC),
                        min_rows=int(min_rows),
                        min_f1_improvement=float(min_delta),
                        actor_label=actor,
                    )
                except (PromotionError, ValueError) as error:
                    st.error(str(error))
                else:
                    st.success("Cohorte sellada. Evalúala con el archivo exacto.")
    cohorts = service.list_cohorts(capability_id)
    if not cohorts:
        return
    cohort_labels = {item.id: f"{item.created_at:%Y-%m-%d} · {item.rows} casos" for item in cohorts}
    selected = st.selectbox("Cohorte sellada", list(cohort_labels), format_func=cohort_labels.get)
    cohort = service.cohort(selected)
    comparison = service.evaluation_for_cohort(cohort.id)
    if comparison is None:
        with st.form(f"evaluate-promotion-{cohort.id}"):
            file = st.file_uploader("Archivo sellado exacto", type=["csv"])
            actor = st.text_input("Quién ejecuta la comparación")
            evaluate = st.form_submit_button("Evaluar ambas versiones")
        if evaluate:
            if file is None:
                st.error("Sube el archivo sellado.")
            else:
                try:
                    service.evaluate(
                        cohort.id, filename=file.name, content=file.getvalue(), actor_label=actor
                    )
                except (PromotionError, ValueError) as error:
                    st.error(str(error))
                else:
                    st.success("Comparación registrada. Revisa los resultados.")
        return
    st.write(
        f"F1 vigente: {comparison.current_f1:.3f} · F1 candidata: "
        f"{comparison.candidate_f1:.3f} · {comparison.recommendation}"
    )
    st.caption(
        f"Escenario neto vigente: {comparison.current_scenario.net_value}; "
        f"candidata: {comparison.candidate_scenario.net_value}. "
        "Ambos usan los mismos supuestos; no son ingresos/ahorros observados."
    )
    evidence_hash = canonical_hash(comparison)
    st.code(evidence_hash, language=None)
    with st.form(f"review-promotion-{comparison.id}"):
        decision = st.radio("Revisión de esta comparación", ["APPROVED", "REJECTED"])
        actor = st.text_input("Quién revisa la evidencia")
        review = st.form_submit_button("Registrar revisión")
    if review:
        try:
            service.review(
                comparison.id, approved_hash=evidence_hash, decision=decision, actor_label=actor
            )
        except (PromotionError, ValueError) as error:
            st.error(str(error))
        else:
            st.success("Revisión registrada.")
    if comparison.recommendation != "CANDIDATE_BETTER":
        st.info("La evidencia no habilita promover la candidata; se conserva el modelo vigente.")
        return
    with st.form(f"promote-{comparison.id}"):
        actor = st.text_input("Quién decide promover")
        confirmed = st.checkbox("Confirmo promover el par modelo-política evaluado")
        promote = st.form_submit_button("Promover candidata")
    if promote:
        if not confirmed:
            st.error("Confirma la decisión humana.")
        else:
            try:
                service.promote(comparison.id, actor_label=actor, approved_hash=evidence_hash)
            except (PromotionError, ValueError) as error:
                st.error(str(error))
            else:
                st.success("Candidata promovida. La anterior queda en el historial.")

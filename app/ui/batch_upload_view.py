"""V2-08/V2-09: subir varios archivos en tres pasos, con los datos comunes una sola vez.

1. Elegir archivos. 2. Datos comunes (origen, licencia, tipo, permisos, responsable y
aprobador) con cambios por archivo si hace falta. 3. Revisar qué se encontró en cada archivo y
subir los aptos; un archivo con problemas no bloquea a los demás y dice qué hacer.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import streamlit as st

from app.ui.formatting import format_number
from core.data_engine.batch import BatchFileV1, BatchIngestionService, SharedAdmissionV1

CLASSIFICATIONS = {
    "synthetic": "Sintético (inventado)",
    "sandbox": "Prueba controlada",
    "anonymized": "Anonimizado",
    "pending_review": "Pendiente de revisión",
}
_STATUS = {
    "accepted": "Apto",
    "rejected": "No apto",
    "duplicate": "Ya estaba en el proyecto",
    "error": "No se pudo revisar",
}
_KEY = "batch-upload"


def _state(project_id: str) -> dict[str, Any]:
    return st.session_state.setdefault(f"{_KEY}-{project_id}", {"step": 1})


def _file_count(count: int) -> str:
    return f"{count} archivo" if count == 1 else f"{count} archivos"


def _is_csv(filename: str) -> bool:
    return Path(filename).suffix.lower() == ".csv"


def render_batch_upload(
    repository: Any,
    storage_root: Path,
    project_id: str,
    *,
    on_done: Callable[[str], None] | None = None,
) -> None:
    state = _state(project_id)
    roles = repository.get_project_roles(project_id)
    step = state["step"]
    st.caption(
        f"Subir archivos · parte {step} de 3: "
        + ("Elige tus archivos", "Datos comunes", "Revisa y sube")[step - 1]
    )

    if step == 1:
        files = st.file_uploader(
            "Archivos de datos (CSV)",
            accept_multiple_files=True,
            key=f"{_KEY}-files-{project_id}",
            help="Puedes elegir varios a la vez. Cada uno: CSV UTF-8, hasta 25 MB, "
            "100.000 filas y 100 columnas.",
        )
        valid_files = [item for item in files if _is_csv(item.name)]
        invalid_files = [item for item in files if not _is_csv(item.name)]
        if valid_files:
            st.markdown("**Se subirán**")
            st.write(", ".join(f"«{item.name}»" for item in valid_files))
        if invalid_files:
            st.markdown("**No se subirán (y por qué)**")
            for item in invalid_files:
                st.warning(
                    f"Este archivo no es una tabla CSV: «{item.name}». "
                    "Sube un archivo .csv (puedes guardarlo así desde Excel con "
                    "«Guardar como → CSV»)."
                )
        if st.button("Continuar", key=f"{_KEY}-next1-{project_id}", type="primary"):
            if not valid_files:
                st.error("Elige al menos un archivo CSV para continuar.")
                return
            state["files"] = [(item.name, item.getvalue()) for item in valid_files]
            state["step"] = 2
            st.rerun()
        return

    names = [name for name, _ in state.get("files", [])]
    if step == 2:
        st.markdown(
            f"Vas a subir **{_file_count(len(names))}**: " + ", ".join(f"«{n}»" for n in names)
        )
        # Recorrido 22/10: volver desde «Revisa y sube» borraba todo lo ya escrito.
        previous = state.get("shared") or {}
        with st.form(f"{_KEY}-shared-{project_id}"):
            st.markdown("**Datos comunes para todos los archivos**")
            source = st.text_input(
                "¿De dónde salieron?",
                value=previous.get("source_name", ""),
                placeholder="Ej.: datos inventados por el equipo para esta prueba",
            )
            license_name = st.text_input(
                "Condiciones de uso (opcional)",
                value=previous.get("license") or "",
                placeholder="Ej.: sólo para esta prueba interna",
            )
            classification = st.selectbox(
                "¿Qué tipo de datos son?",
                list(CLASSIFICATIONS),
                index=(
                    list(CLASSIFICATIONS).index(previous["classification"])
                    if previous.get("classification") in CLASSIFICATIONS
                    else None
                ),
                format_func=CLASSIFICATIONS.get,
                placeholder="Elige una opción",
            )
            authorized = st.checkbox(
                "Confirmo que tengo permiso para usar estos datos en esta prueba",
                value=previous.get("authorization_status") == "approved",
            )
            anonymized = st.checkbox(
                "Confirmo que no tienen datos que identifiquen a personas reales",
                value=bool(previous.get("anonymization_declared")),
            )
            owner = st.text_input(
                "Responsable del proyecto",
                value=previous.get("owner_label")
                or ("" if roles is None else str(roles["owner_label"])),
                placeholder="Nombre o rol",
            )
            approver = st.text_input(
                "Quién aprueba",
                value=previous.get("approver_label")
                or ("" if roles is None else str(roles["approver_label"])),
                placeholder="Nombre o rol",
            )
            overrides: dict[str, str | None] = {}
            if len(names) > 1:
                with st.expander("Cambiar el tipo de datos de un archivo en particular"):
                    for name in names:
                        overrides[name] = st.selectbox(
                            f"«{name}»",
                            [None, *CLASSIFICATIONS],
                            format_func=lambda value: (
                                "Igual que los demás" if value is None else CLASSIFICATIONS[value]
                            ),
                            key=f"{_KEY}-override-{project_id}-{name}",
                        )
            submitted = st.form_submit_button("Revisar los archivos", type="primary")
        if st.button("Volver a elegir archivos", key=f"{_KEY}-back2-{project_id}"):
            state["step"] = 1
            st.rerun()
        if not submitted:
            return
        missing = [
            label
            for label, ok in (
                ("cuenta de dónde salieron", bool(source.strip())),
                ("elige qué tipo de datos son", classification is not None),
                ("escribe el responsable", bool(owner.strip())),
                ("escribe quién aprueba", bool(approver.strip())),
            )
            if not ok
        ]
        if missing:
            st.error("Para continuar, " + ", ".join(missing) + ".")
            return
        state["shared"] = {
            "source_name": source,
            "license": license_name or None,
            "classification": classification,
            "authorization_status": "approved" if authorized else "pending",
            "anonymization_declared": anonymized,
            "owner_label": owner,
            "approver_label": approver,
        }
        state["overrides"] = {name: value for name, value in overrides.items() if value}
        state["step"] = 3
        st.rerun()
        return

    service = BatchIngestionService(repository, storage_root)
    files = [
        BatchFileV1(
            filename=name,
            content=content,
            overrides={"classification": state["overrides"][name]}
            if name in state.get("overrides", {})
            else {},
        )
        for name, content in state["files"]
    ]
    shared = SharedAdmissionV1.model_validate(state["shared"])
    preview = service.preview(files, shared)
    st.markdown("**Esto encontramos en cada archivo**")
    for item in preview.files:
        line = f"**{item.filename}** · {_STATUS[item.status]}"
        if item.rows is not None:
            line += (
                f" · {format_number(item.rows, decimals=0)} filas, "
                f"{format_number(item.columns, decimals=0)} columnas"
            )
        if item.status == "accepted":
            st.success(line)
        else:
            st.warning(f"{line}\n\n**Qué puedes hacer:** {item.message or 'Revisa el archivo.'}")
    ready = sum(1 for item in preview.files if item.status == "accepted")
    columns = st.columns(2)
    if columns[1].button("Volver a los datos comunes", key=f"{_KEY}-back3-{project_id}"):
        state["step"] = 2
        st.rerun()
    if not ready:
        st.info("Ningún archivo está listo todavía. Corrige lo indicado y vuelve a revisar.")
        return
    ready_label = f"{_file_count(ready)} " + ("apto" if ready == 1 else "aptos")
    if columns[0].button(
        f"Subir {ready_label}", key=f"{_KEY}-confirm-{project_id}", type="primary"
    ):
        result = service.ingest(project_id, files, shared)
        accepted = [item.filename for item in result.files if item.status == "accepted"]
        st.session_state.pop(f"{_KEY}-{project_id}", None)
        message = (
            ("Se subió " if len(accepted) == 1 else "Se subieron ")
            + f"{_file_count(len(accepted))}: "
            + ", ".join(f"«{n}»" for n in accepted)
            + "."
        )
        if on_done is not None:
            on_done(message)
        else:
            st.success(message)

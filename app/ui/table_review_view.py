"""V2-11: revisar una tabla antes de usarla para entrenar (aviso, bloqueo y qué se pierde)."""

from typing import Any

import streamlit as st

from core.domain.table_guard import compare_tables


def review_table_change(repository: Any, project_id: str, dataset_id: str) -> str:
    """Muestra la revisión y devuelve «ok», «warning» o «block»."""

    review = compare_tables(repository, project_id, dataset_id)
    lines = [*review.warnings, *review.invalidation_messages]
    if review.level == "block":
        st.error(
            "Esta tabla no sirve para entrenar.\n\n"
            + "\n".join(f"- {line}" for line in review.warnings)
            + "\n\n**Qué puedes hacer:** sigue con la tabla actual o arma una con una fila por "
            "caso desde «Explorar tus datos» → «Preparar una tabla para entrenar»."
        )
    elif lines:
        st.warning(
            "Antes de cambiarla, ten en cuenta:\n\n" + "\n".join(f"- {line}" for line in lines)
        )
    return review.level

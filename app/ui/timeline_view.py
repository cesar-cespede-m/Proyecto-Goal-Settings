"""V2-28: cómo ha ido aprendiendo el modelo, versión por versión."""

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.ui.copy import METRIC_LABELS
from core.domain.learning_timeline import learning_timeline


def render_learning_timeline(repository: Any, storage_root: Path, project_id: str) -> None:
    timeline = learning_timeline(repository, storage_root, project_id)
    if len(timeline.entries) < 1:
        return
    with st.expander("Cómo ha ido aprendiendo el modelo"):
        st.caption(timeline.summary)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Fecha": entry.started_at[:16],
                        "Proyecto": entry.project_name,
                        "Origen": entry.origin,
                        "Medida": METRIC_LABELS.get(entry.metric or "", entry.metric or "—"),
                        "En la prueba final": "—"
                        if entry.test_value is None
                        else round(entry.test_value, 3),
                        "Aprobada para usar": "Sí" if entry.approved else "No",
                        "Usos": entry.uses,
                    }
                    for entry in timeline.entries
                ]
            ),
            hide_index=True,
            width="stretch",
        )

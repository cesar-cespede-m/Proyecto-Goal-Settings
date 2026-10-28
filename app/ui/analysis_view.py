"""Resultado de cualquier análisis: gráfico y tabla (GA-03/GA-04, pantalla).

Genérico: el tipo de gráfico y de eje se deciden por el tipo de cada columna del resultado,
nunca por su nombre. El gráfico siempre va acompañado de su tabla.
"""

import re
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from app.ui.tokens import CHART_SEQUENTIAL, CHART_SERIES, color
from core.analysis.contracts import AnalysisOutputV1, ColumnType
from core.analysis.engine import RESULT_TRUNCATED, AnalysisResultV1

MAX_SERIES = len(CHART_SERIES)
OTHERS_LABEL = "Otros"
_FEW_NUMBERS = 31  # un número con pocos valores distintos (hora, día, mes) se lee por orden


_RANGE_LABEL = re.compile(r"^(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)–(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)$")


def _spanish_range(value: Any) -> Any:
    """«0.9–18.5» se muestra «0,9–18,5»: con punto, «12.022» se lee como doce mil."""

    if not isinstance(value, str) or not (match := _RANGE_LABEL.match(value)):
        return value
    return "–".join(part.replace(".", ",") for part in match.groups())


def result_frame(result: AnalysisResultV1) -> pd.DataFrame:
    """Sólo para mostrar: los tramos numéricos se escriben con coma decimal."""

    frame = pd.DataFrame(result.rows, columns=[column.name for column in result.columns])
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].map(_spanish_range)
    return frame


def _encoding_type(kind: ColumnType, values: pd.Series) -> str:
    if kind is ColumnType.DATETIME:
        return "temporal"
    if kind is ColumnType.NUMBER:
        return "ordinal" if values.nunique() <= _FEW_NUMBERS else "quantitative"
    return "nominal"


def _fold_series(frame: pd.DataFrame, series: str, measure: str) -> pd.DataFrame:
    """Con más series que colores, las de menor peso se agrupan en «Otros»."""

    totals = frame.groupby(series)[measure].sum().sort_values(ascending=False)
    if len(totals) <= MAX_SERIES:
        return frame
    keep = set(totals.index[: MAX_SERIES - 1])
    folded = frame.copy()
    folded[series] = folded[series].where(folded[series].isin(keep), OTHERS_LABEL)
    keys = [column for column in folded.columns if column != measure]
    return folded.groupby(keys, as_index=False, dropna=False)[measure].sum()


def build_chart(result: AnalysisResultV1, output: AnalysisOutputV1) -> alt.Chart | None:
    """Gráfico Altair para el resultado, o None si sólo corresponde la tabla."""

    names = {column.name: column.type for column in result.columns}
    if output.chart == "table" or output.x not in names or output.y not in names:
        return None
    frame = result_frame(result)
    if frame.empty:
        return None
    x_type = _encoding_type(names[output.x], frame[output.x])
    y_type = "quantitative" if names[output.y] is ColumnType.NUMBER else "nominal"
    tooltip = [
        alt.Tooltip(name, type=_encoding_type(kind, frame[name])) for name, kind in names.items()
    ]

    if output.chart == "heatmap":
        chart = (
            alt.Chart(frame)
            .mark_rect(cornerRadius=2, stroke=color("surface"), strokeWidth=2)
            .encode(
                x=alt.X(output.x, type=x_type),
                y=alt.Y(output.y, type=_encoding_type(names[output.y], frame[output.y])),
                color=alt.Color(
                    output.color or output.y,
                    type="quantitative",
                    scale=alt.Scale(range=list(CHART_SEQUENTIAL)),
                ),
                tooltip=tooltip,
            )
        )
    else:
        series = output.color if output.color in names else None
        if series is not None and y_type == "quantitative":
            frame = _fold_series(frame, series, output.y)
        base = alt.Chart(frame)
        if output.chart == "bar":
            mark = base.mark_bar(cornerRadiusEnd=4, stroke=color("surface"), strokeWidth=2)
        elif output.chart == "line":
            mark = base.mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True))
        else:
            mark = base.mark_circle(size=64, opacity=0.85, stroke=color("surface"), strokeWidth=2)
        encodings: dict[str, Any] = {
            # Recorrido 22/10: el gráfico reordenaba alfabéticamente lo que la tabla ordenaba
            # por valor; se respeta el orden del resultado y los ejes se leen sin guiones bajos.
            "x": alt.X(output.x, type=x_type, sort=None, title=_axis_title(output.x)),
            "y": alt.Y(output.y, type=y_type, title=_axis_title(output.y)),
            "tooltip": tooltip,
        }
        if series is not None:
            encodings["color"] = alt.Color(
                series, type="nominal", scale=alt.Scale(range=list(CHART_SERIES))
            )
        else:
            mark = mark.encode(color=alt.value(CHART_SERIES[0]))
        chart = mark.encode(**encodings)

    return (
        chart.properties(height=320)
        .configure(
            background="transparent",
            font="Manrope, Segoe UI, sans-serif",
            # Recorrido 22/10: los ejes decían «1,000»; en español es «1.000».
            locale=alt.Locale(
                number=alt.NumberLocale(
                    decimal=",", thousands=".", grouping=[3], currency=["$", ""]
                )
            ),
        )
        .configure_axis(
            gridColor=color("border"),
            domainColor=color("border"),
            labelColor=color("text-muted"),
            titleColor=color("text"),
            labelFontSize=12,
            titleFontSize=13,
        )
        .configure_legend(labelColor=color("text"), titleColor=color("text"), orient="top")
        .configure_view(stroke=None)
    )


def _axis_title(name: str) -> str:
    return name.replace("_", " ")


def render_analysis_result(result: AnalysisResultV1, output: AnalysisOutputV1) -> None:
    """Gráfico (si corresponde) y siempre la tabla con todas sus cifras."""

    if result.total_rows == 0:
        st.info("El análisis no devolvió casos. Revisa los filtros o la pregunta.")
        return
    chart = build_chart(result, output)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    st.dataframe(result_frame(result), hide_index=True, width="stretch")
    shown = len(result.rows)
    caption = f"{result.total_rows:,} filas en el resultado".replace(",", ".")
    if RESULT_TRUNCATED in result.warnings:
        caption += f"; se muestran las primeras {shown:,}".replace(",", ".")
    st.caption(caption + ".")

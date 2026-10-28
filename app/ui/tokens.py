"""Tokens primitivos y semánticos del frontend Goal Settings."""

# Paleta «Grafito y pizarra» (20/10/2026): estructura en grises, pizarra para acciones y
# selección, ciruela sólo como acento de identidad (nunca botones ni estados) y colores de
# estado reservados para su significado.
PRIMITIVE_COLORS = {
    "gray-0": "#FFFFFF",
    "gray-50": "#F6F7F8",
    "gray-100": "#EFF1F3",
    "gray-150": "#EEF0F2",
    "gray-300": "#D9DCE0",
    "gray-500": "#7D858E",
    "gray-600": "#5F6670",
    "gray-650": "#59616B",
    "gray-900": "#1F2328",
    "slate-100": "#E6EDF3",
    "slate-700": "#34506A",
    "slate-900": "#1F3447",
    "plum-100": "#EEEBF6",
    "plum-700": "#5A4E8C",
    "plum-900": "#3F3566",
    "green-100": "#E6F4EA",
    "green-800": "#1E6B3A",
    "amber-100": "#FFF4CC",
    "amber-800": "#7A5200",
    "red-100": "#FDECEC",
    "red-800": "#A61B1B",
    "bluegray-100": "#EDF1F4",
    "bluegray-800": "#3D4F5C",
}

SEMANTIC_COLORS = {
    "background": "gray-50",
    "surface": "gray-0",
    "surface-subtle": "gray-100",
    "surface-muted": "gray-150",
    "border": "gray-300",
    "text": "gray-900",
    "text-muted": "gray-650",
    # Acciones (botones) y foco.
    "primary": "slate-700",
    "primary-hover": "slate-900",
    "on-primary": "gray-0",
    "primary-container": "slate-100",
    "on-primary-container": "slate-900",
    "secondary": "gray-600",
    "on-secondary-soft": "gray-650",
    # Acento de identidad y orientación: etiquetas de paso, franja de etapa, encabezado.
    "accent": "plum-700",
    "accent-container": "plum-100",
    "on-accent-container": "plum-900",
    # Estados reservados: sólo cuando el mensaje significa eso.
    "success": "green-800",
    "success-container": "green-100",
    "warning": "amber-800",
    "warning-container": "amber-100",
    "error": "red-800",
    "error-container": "red-100",
    "info": "bluegray-800",
    "info-container": "bluegray-100",
    # Selección y bloqueo (stepper, tarjeta de proyecto).
    "surface-selected": "slate-100",
    "on-surface-selected": "slate-900",
    "border-selected": "slate-700",
    "surface-disabled": "gray-150",
    "on-surface-disabled": "gray-600",
    "border-disabled": "gray-500",
    # Acción destructiva: sólo la confirmación final.
    "danger": "red-800",
    "danger-container": "red-100",
}

SPACING = {
    "0": "0",
    "1": "0.25rem",
    "2": "0.5rem",
    "3": "0.75rem",
    "4": "1rem",
    "5": "1.25rem",
    "6": "1.5rem",
    "8": "2rem",
    "10": "2.5rem",
    "12": "3rem",
    "16": "4rem",
    "20": "5rem",
}

SEMANTIC_SPACING = {
    "inline-tight": "2",
    "control-gap": "3",
    "component-inset": "4",
    "card-inset": "6",
    "section-gap": "8",
    "label-gap": "1",
    "panel-inset": "5",
    "stage-gap": "10",
    "page-inline": "12",
    "page-bottom": "20",
    "page-bottom-compact": "12",
    "page-bottom-tablet": "16",
}

TYPE_SCALE = {
    "xs": "0.75rem",
    "sm": "0.875rem",
    "base": "1rem",
    "lg": "1.125rem",
    "xl": "1.25rem",
    "2xl": "1.5rem",
    "3xl": "2rem",
}

SEMANTIC_TYPE = {
    "caption": "xs",
    "supporting": "sm",
    "body": "base",
    "section-title": "xl",
    "page-title": "3xl",
}

RADIUS = {"sm": "0.5rem", "md": "0.75rem", "lg": "1rem", "pill": "999px"}

BORDER_WIDTH = {"thin": "1px", "emphasis": "2px", "accent": "0.35rem"}

LINE_HEIGHT = {"tight": "1.35", "body": "1.5"}

# Tamaño mínimo de objetivos táctiles y del indicador numérico del stepper.
SIZES = {"touch-target": "44px", "step-badge": "2rem"}
LAYOUT = {"content-max": "92rem", "text-max": "62rem", "sidebar-max": "20rem"}
FONT_FAMILY = '"Manrope", "Segoe UI", sans-serif'


def color(token: str) -> str:
    """Resuelve un token semántico a su primitivo hexadecimal."""

    return PRIMITIVE_COLORS[SEMANTIC_COLORS[token]]


def spacing(token: str) -> str:
    """Resuelve un espaciado semántico a la escala primitiva."""

    return SPACING[SEMANTIC_SPACING[token]]


def type_size(token: str) -> str:
    """Resuelve un uso tipográfico a la escala primitiva."""

    return TYPE_SCALE[SEMANTIC_TYPE[token]]


CONTRAST_PAIRS = (
    (color("text"), color("background"), 4.5, "texto principal / fondo"),
    (color("text-muted"), color("background"), 4.5, "texto secundario / fondo"),
    (color("on-primary"), color("primary"), 4.5, "botón primario"),
    (color("on-primary"), color("primary-hover"), 4.5, "botón primario activo"),
    (color("on-primary-container"), color("primary-container"), 4.5, "acento suave"),
    (color("on-secondary-soft"), color("surface"), 4.5, "acción secundaria"),
    (color("success"), color("success-container"), 4.5, "estado correcto"),
    (color("warning"), color("warning-container"), 4.5, "estado requiere acción"),
    (color("error"), color("error-container"), 4.5, "estado de error"),
    (color("info"), color("info-container"), 4.5, "estado informativo"),
    (color("on-surface-selected"), color("surface-selected"), 4.5, "elemento seleccionado"),
    (color("on-surface-disabled"), color("surface-disabled"), 4.5, "elemento bloqueado"),
    (color("danger"), color("danger-container"), 4.5, "acción destructiva"),
    (color("on-primary"), color("border-selected"), 4.5, "indicador del paso actual"),
    (color("accent"), color("surface"), 4.5, "acento sobre superficie"),
    (color("accent"), color("background"), 4.5, "acento sobre fondo"),
    (color("on-accent-container"), color("accent-container"), 4.5, "distintivo de acento"),
    (color("text-muted"), color("surface-subtle"), 4.5, "texto secundario / barra lateral"),
    (color("border-disabled"), color("surface-disabled"), 3.0, "borde de elemento bloqueado"),
    (color("border-selected"), color("surface"), 3.0, "borde de elemento seleccionado"),
)


def streamlit_theme() -> dict[str, str]:
    """Colores del tema nativo de Streamlit (alertas, controles) derivados de los tokens.

    `.streamlit/config.toml` se genera con `scripts/sync_streamlit_theme.py`; una prueba
    impide que ambos se desincronicen.
    """

    return {
        "primaryColor": color("primary"),
        "backgroundColor": color("background"),
        "secondaryBackgroundColor": color("surface-subtle"),
        "textColor": color("text"),
        "redColor": color("error"),
        "redBackgroundColor": color("error-container"),
        "orangeColor": color("warning"),
        "orangeBackgroundColor": color("warning-container"),
        "yellowColor": color("warning"),
        "yellowBackgroundColor": color("warning-container"),
        "greenColor": color("success"),
        "greenBackgroundColor": color("success-container"),
        "blueColor": color("info"),
        "blueBackgroundColor": color("info-container"),
        "violetColor": color("accent"),
        "violetBackgroundColor": color("accent-container"),
        "borderColor": color("border"),
    }


# Series de gráficos (identidad, no estado). Subconjunto de la paleta de referencia validada con
# el script de la guía de visualización (CVD ≥ 19,9; visión normal ≥ 28,1 entre adyacentes).
# Sin verde, ámbar ni rojo: esos tonos están reservados para correcto, advertencia y error. El
# último tono queda bajo 3:1 sobre el fondo, por eso cada gráfico va siempre con su tabla.
CHART_SERIES = ("#2a78d6", "#eb6834", "#6250d6", "#e87ba4")
# Magnitud (mapas de calor): un solo tono de claro a oscuro, desde los tokens de selección.
CHART_SEQUENTIAL = (color("surface-selected"), color("border-selected"))

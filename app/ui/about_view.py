"""V2-06: qué hace Goal Settings, cómo funciona y qué no puede hacer todavía.

Resume el alcance del producto dentro de la interfaz, para que la persona lo sepa antes de
tropezar con un límite.
"""

import streamlit as st

PROMISE = (
    "Define qué decisión puede mejorar los ingresos o reducir costos de tu producto. "
    "Goal Settings te ayuda a revisar datos, evaluar predicciones y explorar el valor de actuar: "
    "nada se entrena ni se usa sin tu aprobación."
)

STEPS = (
    (
        "Tu producto y tu decisión",
        "Registras un objetivo de ingresos o ahorro, la acción posible "
        "y cómo medirías su resultado. "
        "Puedes empezar sin datos y estimar valor con supuestos explícitos.",
    ),
    (
        "Tus datos",
        "Subes archivos CSV; Goal Settings revisa formato, calidad y datos personales.",
    ),
    (
        "Qué quieres anticipar",
        "Dices qué resultado te interesa y qué datos se conocen antes de que ocurra.",
    ),
    (
        "Plan y primera aprobación",
        "Revisas cómo se entrenará y lo apruebas. Recién aquí se puede entrenar.",
    ),
    (
        "Entrenar y entender",
        "Se comparan tres alternativas contra una referencia simple; sólo "
        "se recomienda una si le gana.",
    ),
    (
        "Reporte y segunda aprobación",
        "Lees qué se puede concluir y sus límites, y apruebas usar el modelo.",
    ),
    (
        "Usar y mejorar",
        "Aplicas el modelo a casos nuevos, vigilas su calidad y lo reentrenas con datos nuevos.",
    ),
)

CAN = (
    "Organizar objetivos de ingresos y ahorro con datos, modelos y evaluaciones independientes.",
    "Pedir al equipo de IA que revise evidencia y proponga pasos; se detiene cuando falta "
    "información o una revisión humana.",
    "Explorar escenarios económicos sintéticos y registrar acciones y resultados por caso. "
    "Los escenarios no acreditan ingresos ni ahorros reales.",
    "Archivos CSV de hasta 25 MB, 100.000 filas y 100 columnas; hasta 20 por proyecto.",
    "Anticipar un sí/no, una entre 3 y 20 categorías, o un número.",
    "Probar el modelo al azar o por fechas (lo antiguo aprende, lo reciente prueba).",
    "Analizar con gráficos y preguntarle a tus datos con tus palabras: cada cifra se calcula y "
    "se verifica.",
    "Usar el modelo con casos nuevos, vigilar su calidad y reentrenarlo con la misma receta.",
)

CANNOT = (
    (
        "Datos reales o con personas identificables",
        "Usa datos inventados, de prueba o anonimizados.",
    ),
    ("Excel, PDF, imágenes o audio", "Guarda la planilla como «CSV UTF-8»."),
    (
        "Pronosticar totales futuros (por ejemplo, ventas del próximo mes)",
        "Hoy se anticipa caso por caso; puedes usar la fecha como dato.",
    ),
    (
        "Agrupar sin un resultado conocido",
        "Hoy necesitas una columna con lo que quieres anticipar.",
    ),
    (
        "Explicar causas",
        "Un modelo encuentra relaciones, no causas: úsalo para priorizar, no "
        "para concluir por qué pasa algo.",
    ),
    ("Que la IA decida o entrene sola", "La IA propone y explica; tú confirmas y apruebas."),
)

LATER = (
    "Conectar fuentes externas para recibir datos automáticamente.",
    "Validar el impacto económico atribuible mediante un piloto con grupo de comparación.",
)


def render_about() -> None:
    st.caption(PROMISE)
    with st.expander("Qué hace Goal Settings, cómo funciona y qué no puede hacer"):
        st.markdown("**Cómo funciona**")
        for index, (title, text) in enumerate(STEPS, start=1):
            st.markdown(f"{index}. **{title}.** {text}")
        st.markdown("**Puedes**")
        for item in CAN:
            st.markdown(f"- {item}")
        st.markdown("**Todavía no puedes** (y qué hacer en su lugar)")
        for item, alternative in CANNOT:
            st.markdown(f"- **{item}.** {alternative}")
        st.markdown("**Previsto para más adelante**")
        for item in LATER:
            st.markdown(f"- {item}")

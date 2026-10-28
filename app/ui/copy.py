"""Textos de ayuda de cada campo, escritos en prosa natural (decisión DU-1, 20/10/2026).

Cada texto explica, como lo haría una persona, por qué se pregunta, qué cambia y un ejemplo
integrado en la frase. No se usan plantillas de etiquetas («Qué es», «Para qué sirve»…).
"""

from typing import NamedTuple


class FieldCopy(NamedTuple):
    """Ayuda visible de un campo: una o pocas frases en lenguaje común."""

    text: str


FIELD_COPY: dict[str, FieldCopy] = {
    # Proyecto y archivo
    "project_name": FieldCopy(
        "Un nombre para encontrar este trabajo después; no cambia nada del análisis. "
        "Por ejemplo, «Entregas atrasadas de octubre»."
    ),
    "active_project": FieldCopy(
        "Todo lo que hagas se guarda dentro del proyecto elegido. Cambiar de proyecto no "
        "mezcla sus archivos ni sus decisiones."
    ),
    "csv_file": FieldCopy(
        "Un CSV es una tabla guardada como texto: en Excel o Google Sheets usa «Guardar como» "
        "o «Descargar» y elige «CSV UTF-8». Cada fila es un caso (por ejemplo, un envío) y cada "
        "columna un dato. Hasta 25 MB, 100.000 filas y 100 columnas, y sólo datos de prueba, "
        "inventados o sin datos personales."
    ),
    "source": FieldCopy(
        "Cuenta de dónde salió el archivo y quién te permitió usarlo, por ejemplo «Datos "
        "inventados por el equipo para esta prueba». Queda registrado; no comprueba por sí solo "
        "que el contenido sea seguro."
    ),
    "license": FieldCopy(
        "Si el archivo viene con condiciones de uso, anótalas (por ejemplo «CC BY 4.0»). Si no "
        "las conoces, escribe «pendiente»."
    ),
    "classification": FieldCopy(
        "Elige qué tipo de datos son. «Sintético»: inventados por una herramienta, sin personas "
        "reales. «Prueba controlada»: datos de un entorno de pruebas. «Anonimizado»: datos reales "
        "a los que se quitó todo lo que identifica a alguien. Si no estás seguro, elige "
        "«Pendiente de revisión»: el archivo queda apartado y no se podrá entrenar con él."
    ),
    "authorization": FieldCopy(
        "Marca esta casilla sólo si tienes permiso para usar estos datos en la prueba. Queda "
        "registrado como tu responsabilidad; no reemplaza una autorización legal."
    ),
    "anonymization": FieldCopy(
        "Antes de marcarla, revisa que el archivo no tenga nombres, correos, teléfonos ni otros "
        "datos que identifiquen a una persona. Goal Settings no puede comprobarlo del todo por ti."
    ),
    "active_file": FieldCopy(
        "Esta es la tabla con la que se entrena el modelo. Todo lo que decidas a continuación "
        "se refiere a ella; puedes cambiarla en «Archivos del proyecto»."
    ),
    # Conversación con la IA
    "assistant_conversation": FieldCopy(
        "Cuéntale a la IA, con tus palabras, qué hay en el archivo y qué quieres decidir con él. "
        "Por ejemplo: «Tengo el registro de los casos de los últimos meses y quiero saber con "
        "anticipación cuáles terminarán en un problema». La IA te propondrá respuestas para las "
        "preguntas de abajo; nada queda decidido hasta que tú lo confirmes."
    ),
    "outgoing_preview": FieldCopy(
        "Esto es exactamente lo que saldría hacia el servicio de IA: tu texto y los nombres de "
        "las columnas con su tipo. Nunca se envían filas, valores ni el archivo. Puedes quitar "
        "columnas o seguir sin IA."
    ),
    "decision_actor": FieldCopy(
        "Sólo tu nombre o tu rol, por ejemplo «Ana, jefa de operaciones». Se guarda junto a cada "
        "decisión que confirmes. Lo que quieres lograr se lo cuentas a la IA en el paso siguiente."
    ),
    # Decisiones del contexto
    "objective": FieldCopy(
        "En una frase, ¿qué decisión quieres tomar mejor con este análisis? Por ejemplo, "
        "«Actuar a tiempo en los casos que probablemente terminen mal». La IA puede proponerla; "
        "tú la confirmas."
    ),
    "target_column": FieldCopy(
        "Elige la columna que dice qué pasó al final con cada caso, con sólo dos respuestas "
        "posibles, como «sí» o «no». Es lo que el sistema aprenderá a anticipar."
    ),
    "task_type": FieldCopy(
        "¿Qué forma tiene lo que quieres anticipar? Si es una respuesta de dos opciones (por "
        "ejemplo, sí o no) el modelo avisa; si son varias categorías, elige una; si es un "
        "número, como una cantidad o un tiempo, estima su valor."
    ),
    "positive_class": FieldCopy(
        "De esas dos respuestas, ¿cuál quieres que el sistema te avise antes de que ocurra? "
        "Normalmente es el problema, como «sí, ocurrió». Esta elección siempre es tuya."
    ),
    "prediction_moment": FieldCopy(
        "¿En qué momento usarías el aviso y quién actuaría? Por ejemplo, «Al registrar cada "
        "caso, la persona responsable del área». Sirve para descartar datos que todavía no "
        "existirían en ese momento."
    ),
    "false_positive": FieldCopy(
        "Imagina que el sistema avisa y al final no pasa nada. ¿Qué te cuesta esa falsa alarma? "
        "Por ejemplo, «Dedicamos tiempo a un caso que no lo necesitaba»."
    ),
    "false_negative": FieldCopy(
        "Ahora al revés: el sistema no avisa y el problema sí ocurre. ¿Qué pasa entonces? Por "
        "ejemplo, «El problema ocurre y nadie alcanzó a prevenirlo»."
    ),
    "feature_use": FieldCopy(
        "Para cada columna, decide si el sistema puede mirarla. Excluye los identificadores "
        "(como un número de folio o de registro) y todo lo que sólo se sabe después del "
        "resultado. Si dudas, "
        "elige «No sé»: la columna queda fuera hasta que alguien la revise."
    ),
    "feature_availability": FieldCopy(
        "La pregunta clave es si conocerías ese dato en el momento de decidir. Por ejemplo, "
        "una medición que sólo se registra cuando el resultado ya ocurrió no sirve para "
        "anticiparlo. Si respondes «no» o «no sé», la columna queda fuera."
    ),
    "primary_metric": FieldCopy(
        "¿Qué te importa más: no dejar pasar ningún problema, que los avisos sean certeros, o "
        "un equilibrio? Con esto se elige qué modelo funciona mejor para ti. Si no dejar pasar "
        "problemas es lo más grave, elige la opción que prioriza no dejar pasar casos reales."
    ),
    "metric_rationale": FieldCopy(
        "Explica en una frase por qué elegiste esa medida, por ejemplo «Preferimos algún aviso "
        "de más antes que un problema sin aviso». Queda escrito para quien revise después."
    ),
    "guardrail_mode": FieldCopy(
        "Puedes exigir un mínimo que el resultado debe cumplir, por ejemplo «que al menos 7 de "
        "cada 10 avisos acierten». Si no lo necesitas, elige no exigirlo: también es una "
        "decisión válida."
    ),
    "guardrail_metric": FieldCopy(
        "Elige qué aspecto del resultado no puede quedar por debajo del mínimo, por ejemplo la "
        "proporción de avisos que aciertan."
    ),
    "guardrail_threshold": FieldCopy(
        "Escribe el mínimo como un número entre 0 y 1: 0,70 significa «al menos 7 de cada 10». "
        "No hay un valor correcto para todos los casos."
    ),
    "guardrail_rationale": FieldCopy(
        "Cuenta por qué ese mínimo tiene sentido, por ejemplo «Por debajo de eso tendríamos "
        "demasiadas llamadas innecesarias»."
    ),
    # Estrategia
    "strategy_origin": FieldCopy(
        "La forma de entrenar sigue reglas fijas del producto, iguales para todos los proyectos. "
        "Generarla no la aprueba: eso lo decides tú más abajo."
    ),
    "strategy_features": FieldCopy(
        "Estas son las columnas que el sistema podrá mirar y las que quedarán fuera, según lo "
        "que confirmaste antes. El archivo original no se modifica."
    ),
    "strategy_split": FieldCopy(
        "El archivo se divide en tres partes: una para aprender, otra para comparar modelos y "
        "una última que se guarda para una prueba final, que se usa una sola vez."
    ),
    "strategy_seed": FieldCopy(
        "La división en partes se hace siempre igual, para que el resultado se pueda repetir y "
        "revisar."
    ),
    "preprocessing": FieldCopy(
        "Los datos vacíos se completan y los textos se convierten a números dentro del proceso, "
        "sin tocar tu archivo."
    ),
    "candidate_models": FieldCopy(
        "Se compara una respuesta de referencia muy simple con dos alternativas. Una alternativa "
        "sólo se recomienda si de verdad le gana a la referencia."
    ),
    "decision_threshold": FieldCopy(
        "Cada caso recibe un puntaje y el sistema avisa desde cierto puntaje en adelante. Ese "
        "punto se elige con los datos de comparación, nunca con la prueba final."
    ),
    "runtime_limits": FieldCopy(
        "Cada entrenamiento tiene un tiempo máximo y se hace uno a la vez, para que nada quede "
        "corriendo sin control."
    ),
    "strategy_warnings": FieldCopy(
        "Antes de aprobar, revisa estos puntos sobre tus datos. No significan que el archivo "
        "esté mal, pero pueden cambiar cómo leer el resultado."
    ),
    "strategy_explanation": FieldCopy(
        "Un resumen en palabras simples de lo que se hará y por qué. Es sólo una explicación: "
        "no cambia la estrategia ni la aprueba."
    ),
    "training_decision": FieldCopy(
        "Este es el primer control humano: nada se entrena hasta que apruebes esta versión "
        "exacta. Si algo no te convence, recházala y explica por qué."
    ),
    "strategy_reviewer": FieldCopy("Tu nombre o rol, para que quede claro quién revisó."),
    "strategy_comment": FieldCopy(
        "Deja una nota sobre lo que revisaste. Si rechazas, es obligatoria: cuenta qué debería "
        "cambiar."
    ),
    # Entrenamiento y resultados
    "train_action": FieldCopy(
        "Entrena y compara los modelos con la estrategia que aprobaste. La IA que conversa "
        "contigo no entrena nada; lo hace el motor de aprendizaje del producto."
    ),
    "run_status": FieldCopy(
        "Muestra si el entrenamiento terminó, falló o sigue en curso. Un fallo nunca se presenta "
        "como un resultado válido."
    ),
    "primary_result": FieldCopy(
        "El resultado de la medida que elegiste como principal, primero al comparar y luego en "
        "la prueba final."
    ),
    "translated_metrics": FieldCopy(
        "Distintas formas de mirar los aciertos y errores, por ejemplo «de 10 problemas reales, "
        "detecta 8». Ninguna demuestra por sí sola un ahorro o una causa."
    ),
    "baseline_comparison": FieldCopy(
        "Indica si alguna alternativa le gana a la respuesta de referencia. Si ninguna gana, no "
        "se recomienda ningún modelo, y eso también es un resultado válido."
    ),
    "uncertainty_range": FieldCopy(
        "Como el archivo es una muestra, cada cifra es aproximada. El rango muestra entre qué "
        "valores podría estar realmente."
    ),
    "confusion_matrix": FieldCopy(
        "Cuenta los aciertos y los dos tipos de error en la prueba final: avisos de más y "
        "problemas que pasaron sin aviso."
    ),
    "feature_importance": FieldCopy(
        "Muestra qué columnas influyen más en la predicción. Indica asociación, no causa: que "
        "el clima influya no prueba que el clima cause el atraso."
    ),
    # Reporte
    "generate_report": FieldCopy(
        "Arma un reporte con las cifras calculadas, lo que se puede concluir y sus límites. La "
        "IA puede redactarlo, pero no puede cambiar ninguna cifra."
    ),
    "generation_mode": FieldCopy(
        "Indica si el texto lo redactó la IA o una plantilla fija. En ambos casos las cifras "
        "son las mismas y están verificadas."
    ),
    "insight_kind": FieldCopy(
        "Cada hallazgo indica si es un dato medido, una lectura, una hipótesis o una "
        "recomendación, para no confundir evidencia con opinión."
    ),
    "evidence": FieldCopy(
        "Cada afirmación del reporte apunta a la cifra que la respalda, para que puedas "
        "comprobarla."
    ),
    "report_decision": FieldCopy(
        "Este es el segundo control humano. Aprueba sólo si los hallazgos y sus límites te "
        "parecen correctos. Aprobar todavía no lo publica."
    ),
    "report_reviewer": FieldCopy("Tu nombre o rol, para que quede claro quién revisó el reporte."),
    "report_comment": FieldCopy(
        "Deja una nota sobre tu revisión. Si rechazas, es obligatoria: cuenta qué debería "
        "corregirse."
    ),
    "publish_action": FieldCopy(
        "Hace visible el reporte aprobado dentro de esta demo privada. No se publica nada en "
        "Internet."
    ),
    "compare_versions": FieldCopy(
        "Si probaste distintas decisiones, aquí ves qué resultado tuvo cada versión. Una "
        "diferencia no prueba que ese cambio la haya causado."
    ),
    "compare_selection": FieldCopy(
        "Elige dos versiones para verlas lado a lado. No se vuelve a entrenar nada."
    ),
    "results_question": FieldCopy(
        "Pregunta lo que quieras entender de estas cifras, por ejemplo «¿Esto me sirve para "
        "actuar antes de que ocurra?». La respuesta se verifica contra las cifras y no se guarda."
    ),
    "report_download": FieldCopy(
        "Descarga el reporte como página web para abrirlo, compartirlo dentro de tu equipo o "
        "imprimirlo como PDF."
    ),
    # Uso con casos nuevos
    "new_cases": FieldCopy(
        "Usa el modelo aprobado para ordenar casos nuevos de mayor a menor riesgo, sin volver a "
        "entrenar. Sirve para priorizar; la decisión final sigue siendo de una persona."
    ),
    "new_cases_file": FieldCopy(
        "Sube un CSV con las mismas columnas que usó el modelo. Si ya conoces el resultado real "
        "de esos casos, inclúyelo y verás cómo le fue. Las filas no se guardan."
    ),
    "scoring_actor": FieldCopy(
        "Tu nombre o rol, para registrar quién usó el modelo con este lote."
    ),
}


GLOSSARY: dict[str, str] = {
    "target": "resultado que quieres anticipar",
    "positive_class": "evento que quieres detectar",
    "recall": "de los eventos reales, cuántos detecta",
    "precision": "de las alertas emitidas, cuántas aciertan",
    "f1": "equilibrio entre detectar y acertar",
    "average_precision": "calidad global de las alertas ordenadas por riesgo",
    "split": "reparto de datos para aprender / comparar / prueba final",
    "train": "datos para aprender",
    "validation": "datos para comparar",
    "test": "prueba final reservada",
    "seed": "número que permite repetir el mismo reparto",
    "holdout": "prueba final reservada",
    "leakage": "información que todavía no existía al tomar la decisión",
    "guardrail": "condición mínima para aceptar un resultado",
    "threshold": "punto de corte para convertir un puntaje en alerta",
    "baseline": "modelo simple de referencia",
    "run": "ejecución",
}


METRIC_LABELS: dict[str, str] = {
    "recall": GLOSSARY["recall"],
    "precision": GLOSSARY["precision"],
    "f1": GLOSSARY["f1"],
    "average_precision": GLOSSARY["average_precision"],
    "balanced_accuracy": "acierto equilibrado entre ambos resultados",
    "roc_auc": "capacidad global para ordenar ambos resultados",
    "macro_f1": "acierto parejo en todas las categorías",
    "mae": "error promedio en la misma unidad del resultado",
    "rmse": "error que castiga más las equivocaciones grandes",
}


# Nombres de los modelos en palabras; el nombre técnico queda en «Ver detalle técnico».
MODEL_NAMES: dict[str, str] = {
    "DummyClassifier": "referencia simple",
    "DummyRegressor": "referencia simple",
    "LogisticRegression": "alternativa lineal",
    "Ridge": "alternativa lineal",
    "RandomForestClassifier": "alternativa de reglas combinadas",
    "RandomForestRegressor": "alternativa de reglas combinadas",
}


def model_name(value: str | None) -> str:
    return "sin modelo recomendado" if value is None else MODEL_NAMES.get(value, value)


def decimal(value: float, digits: int = 3) -> str:
    """Decimales con coma, como en el resto de la app."""

    return f"{value:.{digits}f}".replace(".", ",")

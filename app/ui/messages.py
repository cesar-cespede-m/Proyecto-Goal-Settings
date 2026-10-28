"""Traducción de fallos técnicos a mensajes accionables para la interfaz."""

from difflib import get_close_matches
from typing import NamedTuple

import streamlit as st

from app.ui.incidents import record_incident, render_incident


class CommonError(NamedTuple):
    message: str
    action: str


def _missing_feature_columns_message(error: BaseException) -> CommonError | None:
    """Nombra columnas ausentes y sugiere encabezados parecidos, si están disponibles."""

    missing = [str(column) for column in getattr(error, "missing_columns", ())]
    available = [str(column) for column in getattr(error, "available_columns", ())]
    if not missing:
        return None

    suggestions = {
        column: matches[0]
        for column in missing
        if (matches := get_close_matches(column, available, n=1, cutoff=0.6))
    }
    if len(missing) == 1:
        column = missing[0]
        suggestion = suggestions.get(column)
        if suggestion:
            return CommonError(
                f"Falta «{column}». ¿Es «{suggestion}»?",
                "Cámbiale el nombre y vuelve a subirlo.",
            )
        return CommonError(
            f"Falta «{column}».",
            "Añade esa columna con el mismo nombre usado durante el aprendizaje y vuelve a "
            "subirlo.",
        )

    columns = ", ".join(f"«{column}»" for column in missing)
    possible_names = " ".join(
        f"Para «{column}», ¿es «{suggestion}»?" for column, suggestion in suggestions.items()
    )
    return CommonError(
        f"Faltan estas columnas: {columns}." + (f" {possible_names}" if possible_names else ""),
        (
            "Cambia los nombres sugeridos o añade las columnas que faltan y vuelve a subirlo."
            if suggestions
            else "Añade esas columnas con los mismos nombres usados durante el aprendizaje y "
            "vuelve a subirlo."
        ),
    )


ERROR_MESSAGES: dict[str, CommonError] = {
    "TARGET_NOT_BINARY": CommonError(
        "El resultado elegido no tiene exactamente dos valores válidos.",
        "Elige otra columna o corrige el archivo para que tenga dos resultados sin vacíos.",
    ),
    "TARGET_HAS_MISSING_VALUES": CommonError(
        "El resultado que quieres anticipar contiene valores vacíos.",
        "Completa o elimina esos casos en una copia segura del archivo y vuelve a cargarla.",
    ),
    "INVALID_CONTEXT": CommonError(
        "No pudimos guardar estas decisiones porque falta información "
        "o hay opciones incompatibles.",
        "Revisa los campos marcados, responde cada decisión y vuelve a intentarlo.",
    ),
    "CONTEXT_UNCHANGED": CommonError(
        "Estas decisiones son iguales a la versión que ya está guardada.",
        "No necesitas crear otra versión; cambia un dato sólo si la decisión realmente cambió.",
    ),
    "PROJECT_DATASET_MISMATCH": CommonError(
        "El archivo seleccionado no pertenece al proyecto activo.",
        "Vuelve a seleccionar el proyecto y su archivo antes de continuar.",
    ),
    "DATASET_NOT_ACCEPTED": CommonError(
        "Este archivo todavía no está aceptado para el experimento.",
        "Revisa su clasificación, autorización y anonimización o carga otro archivo permitido.",
    ),
    "DATASET_FILE_UNAVAILABLE": CommonError(
        "No encontramos la copia privada del archivo.",
        "Vuelve a cargar el CSV permitido o pide revisar el almacenamiento de la demo.",
    ),
    "DATASET_PARSE_FAILED": CommonError(
        "No pudimos leer el contenido como un CSV válido.",
        "Guárdalo como CSV UTF-8, comprueba sus columnas y vuelve a cargarlo.",
    ),
    "FILE_TOO_LARGE": CommonError(
        "El archivo supera el límite de 25 MB.",
        "Usa una muestra sintética o autorizada más pequeña y vuelve a cargarla.",
    ),
    "TOO_MANY_ROWS": CommonError(
        "El archivo supera el límite de 100.000 filas.",
        "Reduce la muestra permitida a 100.000 filas o menos.",
    ),
    "TOO_MANY_COLUMNS": CommonError(
        "El archivo supera el límite de 100 columnas.",
        "Conserva sólo las columnas necesarias en una copia segura y vuelve a cargarla.",
    ),
    "DUPLICATE_DATASET": CommonError(
        "Este mismo archivo ya está registrado en el proyecto.",
        "Selecciona el archivo existente o carga una versión realmente diferente.",
    ),
    "DATASET_ALREADY_REGISTERED": CommonError(
        "Este mismo archivo ya está registrado en el proyecto.",
        "Selecciona la fuente existente o carga una versión realmente diferente.",
    ),
    "PROJECT_SOURCE_LIMIT_REACHED": CommonError(
        "Este proyecto ya tiene el máximo de cinco fuentes CSV.",
        "Compón las fuentes disponibles o crea otro proyecto para un análisis diferente.",
    ),
    "COMPOSITION_ACTOR_REQUIRED": CommonError(
        "Falta indicar quién revisó el cruce.",
        "Escribe el nombre o rol de la persona responsable antes de previsualizar.",
    ),
    "JOIN_KEY_HAS_MISSING": CommonError(
        "Una columna elegida para relacionar archivos contiene valores vacíos.",
        "Elige otra clave o completa esos valores en una copia segura del CSV.",
    ),
    "JOIN_KEY_TYPE_MISMATCH": CommonError(
        "Las columnas elegidas para relacionar archivos no tienen el mismo tipo.",
        "Usa claves equivalentes en ambos CSV, por ejemplo números con números o texto con texto.",
    ),
    "AUXILIARY_KEY_NOT_UNIQUE": CommonError(
        "La clave del archivo auxiliar aparece más de una vez.",
        "Usa una columna única o consolida el auxiliar antes de volver a cruzarlo.",
    ),
    "PRIMARY_KEY_NOT_UNIQUE": CommonError(
        "La clave principal tiene duplicados y no cumple una relación uno a uno.",
        "Elige la relación de varias filas principales por una auxiliar o usa una clave única.",
    ),
    "COMPOSITION_INPUT_TOO_LARGE": CommonError(
        "Las fuentes seleccionadas superan el límite agregado de 50 MB.",
        "Reduce las muestras permitidas o selecciona menos fuentes.",
    ),
    "COMPOSED_TOO_MANY_COLUMNS": CommonError(
        "El cruce produciría más de 100 columnas.",
        "Reduce las columnas de las fuentes auxiliares antes de cargarlas.",
    ),
    "INVALID_DATASET_COMPOSITION": CommonError(
        "No pudimos crear este cruce con la configuración actual.",
        "Revisa las columnas de unión y el responsable; si ese cruce ya existe, sigue con el "
        "archivo activo del proyecto.",
    ),
    "INVALID_STRATEGY": CommonError(
        "No pudimos preparar una estrategia válida con estas decisiones.",
        "Revisa el contexto y las advertencias antes de volver a generarla.",
    ),
    "STRATEGY_CONTEXT_MISMATCH": CommonError(
        "La estrategia ya no corresponde a las decisiones vigentes.",
        "Genera una estrategia nueva para el contexto actual.",
    ),
    "STRATEGY_UNCHANGED": CommonError(
        "La estrategia fija ya existe y no cambió.",
        "Revisa la versión vigente en lugar de crear otra.",
    ),
    "STRATEGY_NOT_CURRENT": CommonError(
        "La estrategia que intentas revisar ya no es la vigente.",
        "Recarga la pantalla y decide sobre la versión más reciente.",
    ),
    "STRATEGY_NOT_APPROVED": CommonError(
        "El entrenamiento está bloqueado porque falta una aprobación vigente.",
        "Revisa la estrategia actual y registra una decisión explícita.",
    ),
    "TRAINING_TIME_CAP_EXCEEDED": CommonError(
        "El aprendizaje excedió el tiempo máximo permitido.",
        "Revisa el tamaño del archivo y las advertencias antes de iniciar otra ejecución.",
    ),
    "CANDIDATE_TRAINING_FAILED": CommonError(
        "Uno de los modelos no pudo completar el aprendizaje.",
        "Revisa el archivo y las advertencias; el fallo no cuenta como resultado válido.",
    ),
    "DATASET_HASH_MISMATCH": CommonError(
        "El archivo guardado ya no coincide con la versión que se aprobó.",
        "Vuelve a cargar el archivo correcto y repite la revisión desde el contexto.",
    ),
    "RUN_NOT_REPORTABLE": CommonError(
        "La ejecución no terminó en un estado que permita crear un reporte.",
        "Revisa el fallo de entrenamiento e inicia una nueva ejecución cuando esté resuelto.",
    ),
    "REPORT_NOT_APPROVED": CommonError(
        "El reporte todavía no tiene una aprobación vigente.",
        "Revísalo, registra una decisión explícita y luego usa la acción de publicar.",
    ),
    "INVALID_REPORT": CommonError(
        "No pudimos crear un reporte verificable con este resultado.",
        "Revisa el estado de la ejecución y vuelve a generarlo.",
    ),
    "INVALID_STATE_TRANSITION": CommonError(
        "Esta acción no está disponible en el estado actual del flujo.",
        "Recarga la pantalla y completa el paso anterior indicado.",
    ),
    "EMPTY_FILE": CommonError(
        "El archivo está vacío.",
        "Elige un CSV que contenga encabezados y al menos una fila de datos.",
    ),
    "AMBIGUOUS_DELIMITER": CommonError(
        "El archivo no usa comas para separar sus columnas.",
        "Vuelve a guardarlo como CSV separado por comas.",
    ),
    "EMPTY_HEADER": CommonError(
        "Una o más columnas no tienen nombre.",
        "Asigna un nombre único a cada columna y vuelve a cargar el archivo.",
    ),
    "DUPLICATE_HEADERS": CommonError(
        "Hay nombres de columna repetidos.",
        "Cambia los encabezados para que cada columna tenga un nombre único.",
    ),
    "MALFORMED_ROW": CommonError(
        "Una fila tiene una cantidad de campos distinta a los encabezados.",
        "Corrige la estructura del CSV y vuelve a cargarlo.",
    ),
    "MALFORMED_CSV": CommonError(
        "No pudimos interpretar la estructura del CSV.",
        "Comprueba comillas, separadores y saltos de línea antes de volver a cargarlo.",
    ),
    "NO_DATA_ROWS": CommonError(
        "El archivo no contiene filas de datos.",
        "Carga un CSV permitido con al menos una fila además de los encabezados.",
    ),
    "INVALID_EXTENSION": CommonError(
        "El archivo no tiene extensión CSV.",
        "Guarda una copia como .csv y vuelve a seleccionarla.",
    ),
    "COMPRESSED_FILE": CommonError(
        "El archivo está comprimido y no puede analizarse.",
        "Extrae el CSV permitido y carga el archivo sin comprimir.",
    ),
    "INVALID_ENCODING": CommonError(
        "El texto del archivo no está codificado como UTF-8.",
        "Guarda una copia como CSV UTF-8 y vuelve a cargarla.",
    ),
    "BINARY_CONTENT": CommonError(
        "El archivo contiene datos binarios que no corresponden a un CSV.",
        "Selecciona el CSV de texto correcto.",
    ),
    "POSSIBLE_PERSONAL_DATA": CommonError(
        "El archivo contiene columnas que podrían identificar personas.",
        "Elimina esas columnas o carga una copia debidamente anonimizada.",
    ),
    "BATCH_INGESTION_FAILED": CommonError(
        "No pudimos procesar uno de los archivos del lote.",
        "Vuelve a seleccionarlo; si el problema continúa, comparte el código del incidente.",
    ),
    "AUTHORIZATION_PENDING": CommonError(
        "La autorización del archivo todavía está pendiente.",
        "Obtén la autorización de la fuente y vuelve a confirmar la carga.",
    ),
    "AUTHORIZATION_REJECTED": CommonError(
        "La autorización declarada para este archivo fue rechazada.",
        "Selecciona otra fuente que tenga una autorización vigente.",
    ),
    "CLASSIFICATION_PENDING_REVIEW": CommonError(
        "La clasificación del archivo todavía necesita revisión.",
        "Confirma que es una fuente sintética, de prueba o debidamente anonimizada.",
    ),
    "ANONYMIZATION_NOT_DECLARED": CommonError(
        "Falta declarar la anonimización de este archivo.",
        "Verifica la copia anonimizada y confirma la declaración antes de cargarla.",
    ),
    "PARSING_FAILED": CommonError(
        "No pudimos convertir el CSV validado en una tabla.",
        "Revisa su estructura y vuelve a exportarlo como CSV UTF-8.",
    ),
    "TARGET_NOT_FOUND": CommonError(
        "La columna elegida ya no está disponible en el archivo.",
        "Recarga la pantalla y elige uno de los resultados disponibles.",
    ),
    "TARGET_HAS_MISSING": CommonError(
        "El resultado que quieres anticipar tiene valores vacíos.",
        "Completa o elimina esos casos en una copia segura y vuelve a cargarla.",
    ),
    "POSITIVE_CLASS_NOT_FOUND": CommonError(
        "El evento elegido ya no coincide con los valores del archivo.",
        "Vuelve a elegir el resultado y luego selecciona uno de sus dos eventos.",
    ),
    "FEATURE_DECISIONS_INCOMPLETE": CommonError(
        "Falta decidir el uso o la disponibilidad de alguna variable.",
        "Responde cada variable antes de guardar.",
    ),
    "FEATURE_NOT_FOUND": CommonError(
        "Una variable de estas decisiones ya no existe en el archivo.",
        "Recarga la pantalla y revisa nuevamente todas las variables.",
    ),
    "NO_ALLOWED_FEATURES": CommonError(
        "No quedó ninguna variable disponible para aprender.",
        "Revisa si al menos una variable segura existe en el momento de decidir.",
    ),
    "UNSAFE_FEATURE_INCLUDED": CommonError(
        "Se intentó usar un identificador o una variable constante.",
        "Mantén esas variables excluidas y revisa las demás.",
    ),
    "INSUFFICIENT_CLASS_SUPPORT": CommonError(
        "Hay muy pocos casos de uno de los resultados para repartirlos de forma segura.",
        "Usa un archivo permitido con más ejemplos de ambos resultados.",
    ),
    "UNSUPPORTED_TARGET_VALUE": CommonError(
        "El resultado contiene un tipo de valor que Goal Settings no admite.",
        "Convierte sus dos valores a texto, número o verdadero/falso y vuelve a cargarlo.",
    ),
    "PROJECT_CONTEXT_MISMATCH": CommonError(
        "Estas decisiones no pertenecen al proyecto activo.",
        "Recarga la pantalla y selecciona el proyecto correcto.",
    ),
    "CONTEXT_NOT_CURRENT": CommonError(
        "Hay una versión más reciente de las decisiones.",
        "Revisa la versión vigente y genera de nuevo la estrategia.",
    ),
    "TRAINING_ALREADY_RUNNING": CommonError(
        "Ya hay una ejecución en curso.",
        "Espera a que termine antes de iniciar otra.",
    ),
    "TRAINING_FAILED": CommonError(
        "El aprendizaje no pudo completarse.",
        "Revisa el archivo, la estrategia y el detalle técnico antes de intentarlo otra vez.",
    ),
    "NO_FEATURES": CommonError(
        "No hay variables disponibles para aprender.",
        "Vuelve al contexto y confirma al menos una variable segura.",
    ),
    "EVALUATION_SCHEMA_UNSUPPORTED": CommonError(
        "La ejecución usa una versión anterior que ya no puede reportarse.",
        "Inicia una nueva ejecución con la estrategia vigente.",
    ),
    "UNVERIFIED_CLAIMS": CommonError(
        "El reporte contiene una afirmación que no pudo vincularse a evidencia.",
        "Genera otra versión o usa la plantilla verificada.",
    ),
    "PROJECT_ALREADY_HAS_DATASET": CommonError(
        "Este proyecto ya tiene su archivo aceptado; cada proyecto trabaja con uno solo.",
        "Para analizar otro archivo, crea un proyecto nuevo.",
    ),
    "REPORT_NOT_CURRENT": CommonError(
        "Este reporte corresponde a una versión superada del contexto o la estrategia.",
        "Puedes consultarlo o rechazarlo, pero no aprobarlo ni publicarlo. "
        "Entrena de nuevo con la versión vigente.",
    ),
    "DRAFT_CHANGED_ELSEWHERE": CommonError(
        "El borrador cambió en otra pestaña o sesión.",
        "Recarga la página para ver la versión más reciente antes de seguir.",
    ),
    "DRAFT_NOT_ACTIVE": CommonError(
        "Este borrador ya se cerró.",
        "Recarga la página; se abrirá un borrador nuevo.",
    ),
    "UNKNOWN_OUTCOME_VALUES": CommonError(
        "Algunas filas traen un resultado real distinto de los dos valores con que se entrenó.",
        "Corrige o elimina esas filas para medir cómo le fue al modelo, o quita la columna del "
        "resultado para sólo priorizar.",
    ),
    "ACTOR_REQUIRED": CommonError(
        "Falta indicar quién usa el modelo.",
        "Escribe tu nombre o rol; queda en el registro de auditoría.",
    ),
    "DRAFT_DATASET_CHANGED": CommonError(
        "El archivo cambió desde que empezaste el borrador.",
        "Descarta el borrador y vuelve a empezar con el archivo vigente.",
    ),
    "RUN_HAS_NO_MODEL": CommonError(
        "Esta ejecución no terminó con un modelo recomendado que pueda usarse.",
        "Revisa los resultados o ejecuta una estrategia vigente que supere la referencia.",
    ),
    "MODEL_NOT_APPROVED_FOR_USE": CommonError(
        "El modelo todavía no está aprobado para priorizar casos nuevos.",
        "Revisa el reporte y registra la segunda aprobación humana.",
    ),
    "MODEL_NOT_CURRENT": CommonError(
        "El modelo pertenece a decisiones o a una estrategia que ya fueron reemplazadas.",
        "Entrena y aprueba un reporte con la versión vigente antes de usarlo.",
    ),
    "MODEL_NOT_ACTIVE": CommonError(
        "Esta candidata aún no es el modelo vigente de la capacidad.",
        "Compara ambas versiones y completa la revisión humana antes de promoverla.",
    ),
    "ARTIFACT_INTEGRITY_FAILED": CommonError(
        "El archivo interno del modelo no coincide con el que se aprobó.",
        "No lo uses; repite el entrenamiento y la revisión del reporte.",
    ),
    "RUN_HAS_NO_THRESHOLD": CommonError(
        "La ejecución no dejó un punto de corte válido para emitir alertas.",
        "Revisa la estrategia y ejecuta nuevamente el aprendizaje.",
    ),
    "MISSING_FEATURE_COLUMNS": CommonError(
        "Al CSV le faltan campos que el modelo necesita para calcular prioridades.",
        "Añade los mismos campos usados durante el aprendizaje y vuelve a intentarlo.",
    ),
    "NON_NUMERIC_FEATURE_VALUES": CommonError(
        "Algunas columnas que el modelo usa como números traen texto, por ejemplo «n/d».",
        "Deja esas celdas vacías o con un número y vuelve a cargar el archivo; las vacías se "
        "completan solas.",
    ),
    "JOIN_HAS_NO_MATCHES": CommonError(
        "Los dos archivos no comparten ningún caso: ninguna clave del principal aparece en el "
        "auxiliar.",
        "Cruzar sólo agrega columnas a los mismos casos. Si querías sumar más casos, sube el "
        "archivo más grande como archivo del proyecto; si no, elige otra columna para unirlos.",
    ),
    "JOIN_KEYS_FORMAT_MISMATCH": CommonError(
        "Las claves parecen las mismas, pero están escritas de otra forma en cada archivo "
        "(mayúsculas, guiones o espacios).",
        "Corrige el formato de la clave en uno de los archivos para que coincidan exactamente.",
    ),
    "DATA_AUTHORIZATION_REQUIRED": CommonError(
        "Falta confirmar que el archivo reciente se puede usar.",
        "Marca la confirmación: sólo se aceptan datos sintéticos, de sandbox o anonimizados.",
    ),
    "RETRAINING_DATASET_NOT_ACCEPTED": CommonError(
        "El archivo reciente no quedó aceptado para entrenar.",
        "Revisa su formato y procedencia y vuelve a intentarlo.",
    ),
    "INVALID_DATASET": CommonError(
        "No pudimos aceptar este archivo.",
        "Revisa que sea un CSV separado por comas, en UTF-8, con encabezados y al menos una fila "
        "de datos, y vuelve a subirlo.",
    ),
    "PROJECT_ARCHIVE_FAILED": CommonError(
        "No pudimos eliminar este proyecto de la lista.",
        "Puede que ya se haya eliminado en otra pestaña: recarga la página y revisa la lista.",
    ),
    "MODELING_TABLE_FAILED": CommonError(
        "No pudimos cambiar la tabla para entrenar.",
        "Recarga la página y elige un archivo aceptado de este proyecto.",
    ),
    "ANALYSIS_FAILED": CommonError(
        "No pudimos completar este análisis.",
        "Revisa las columnas elegidas y vuelve a intentarlo.",
    ),
    "SNAPSHOT_SPEC_INVALID": CommonError(
        "No pudimos interpretar la receta para construir la tabla por caso.",
        "Revisa el corte, las ventanas, las variables y el resultado antes de intentarlo otra vez.",
    ),
    "SNAPSHOT_COLUMN_NOT_FOUND": CommonError(
        "Una columna de la receta no aparece en los archivos seleccionados.",
        "Comprueba las columnas de entidad, momento, variables y resultado.",
    ),
    "SNAPSHOT_TYPE_MISMATCH": CommonError(
        "Una columna no tiene el tipo que necesita el cálculo solicitado.",
        "Usa fechas para el momento y números para sumas, promedios o máximos.",
    ),
    "SNAPSHOT_STAGE_OUT_OF_ORDER": CommonError(
        "Los eventos contienen una etapa que no está en el orden declarado.",
        "Agrega la etapa faltante al orden o corrige ese valor en una copia segura del archivo.",
    ),
    "SNAPSHOT_ENTITY_TABLE_REQUIRED": CommonError(
        "La receta pide atributos fijos, pero falta la tabla que los contiene.",
        "Selecciona la tabla de entidades o quita esos atributos de la receta.",
    ),
    "SNAPSHOT_ENTITY_NOT_UNIQUE": CommonError(
        "La tabla de atributos tiene casos repetidos o identificadores vacíos.",
        "Conserva una sola fila completa por caso y vuelve a intentarlo.",
    ),
    "SNAPSHOT_ENTITY_NOT_FOUND": CommonError(
        "Hay casos de la actividad sin una fila en la tabla de atributos.",
        "Completa la tabla de atributos o usa sólo casos presentes en ambos archivos.",
    ),
    "SNAPSHOT_OUTPUT_COLLISION": CommonError(
        "Dos variables de la receta producirían el mismo nombre de columna.",
        "Asigna nombres de salida distintos a cada variable y resultado.",
    ),
    "SNAPSHOT_EMPTY_EVENTS": CommonError(
        "La tabla de actividad no contiene registros para construir casos.",
        "Selecciona un archivo con al menos un evento y vuelve a intentarlo.",
    ),
    "ANALYSIS_TIME_LIMIT_EXCEEDED": CommonError(
        "El análisis tardó más de lo permitido.",
        "Agrupa por menos columnas o elige un archivo más pequeño.",
    ),
    "ANALYSIS_INPUT_ROW_LIMIT_EXCEEDED": CommonError(
        "El archivo tiene más filas de las que el análisis puede leer de una vez.",
        "Usa un archivo más pequeño o una parte de los datos.",
    ),
    "ANALYSIS_TABLE_FAILED": CommonError(
        "No pudimos guardar el cálculo como tabla.",
        "Vuelve a hacer el cálculo y guárdalo otra vez; tus archivos no cambiaron.",
    ),
    "TABLE_NAME_REQUIRED": CommonError(
        "La tabla necesita un nombre.",
        "Escribe un nombre corto que te ayude a reconocerla en los archivos del proyecto.",
    ),
    "TABLE_ACTOR_REQUIRED": CommonError(
        "Falta indicar quién guarda la tabla.",
        "Escribe tu nombre o rol; queda en el registro de auditoría.",
    ),
    "TABLE_TOO_LARGE": CommonError(
        "El resultado tiene demasiadas filas para guardarlo como tabla.",
        "Filtra los casos o usa un archivo más pequeño y vuelve a intentarlo.",
    ),
    "TABLE_EMPTY": CommonError(
        "El cálculo no dejó ninguna fila para guardar.",
        "Revisa los filtros del cálculo: quizá dejaron fuera todos los casos.",
    ),
    "TABLE_NOT_VALID": CommonError(
        "El resultado no forma una tabla que se pueda guardar.",
        "Revisa que tenga al menos dos filas y nombres de columna distintos.",
    ),
    "UNSUPPORTED_TASK_KIND": CommonError(
        "Este tipo de modelo todavía no se puede entrenar desde el flujo con aprobaciones.",
        "Por ahora usa un resultado sí/no; las varias clases y las cantidades llegan pronto.",
    ),
    "USE_SCORE_FOR_BINARY": CommonError(
        "Este modelo es de sí/no y se usa para priorizar casos, no para estimar valores.",
        "Recarga la página: verás la opción «Priorizar casos» de este modelo.",
    ),
    "RECIPE_NOT_AVAILABLE": CommonError(
        "Este modelo se entrenó con un archivo directo, no con una tabla preparada.",
        "Reentrena subiendo una versión reciente de ese archivo.",
    ),
    "RECIPE_FILES_MISSING": CommonError(
        "Falta la versión nueva de algún archivo de la receta.",
        "Sube un archivo por cada nombre que pide el formulario.",
    ),
    "DECISION_RECORD_FAILED": CommonError(
        "No pudimos guardar la decisión.",
        "Revisa que tenga texto y responsable, y vuelve a intentarlo.",
    ),
    "INVALID_DECISION_RECORD": CommonError(
        "A la decisión le falta información.",
        "Escribe qué decidiste (al menos unas palabras) y quién la toma.",
    ),
    "INVALID_DECISION_OUTCOME": CommonError(
        "Al resultado le falta información.",
        "Describe qué pasó después y quién lo observó.",
    ),
    "DECISION_NOT_FOUND": CommonError(
        "No encontramos esa decisión.",
        "Recarga la página y vuelve a intentarlo.",
    ),
    "INVALID_ACTION_LIST": CommonError(
        "No pudimos registrar esta lista de acción.",
        "Revisa los casos seleccionados, el grupo de control y quién confirma la acción.",
    ),
    "ACTION_LIST_RUN_MISMATCH": CommonError(
        "La lista de acción no corresponde a un modelo de este proyecto.",
        "Vuelve al proyecto correcto y genera la lista desde su modelo vigente.",
    ),
    "ACTION_LIST_NOT_FOUND": CommonError(
        "No encontramos la lista de acción que quieres medir.",
        "Recarga el proyecto y elige una lista guardada antes de registrar resultados.",
    ),
    "INVALID_ACTION_OUTCOME": CommonError(
        "No pudimos registrar los resultados de esta lista.",
        "Indica el resultado que cuenta como evento y quién revisó la medición.",
    ),
    "BUNDLE_INVALID": CommonError(
        "El archivo no es una receta exportada por Goal Settings.",
        "Exporta la receta desde «Usar el modelo» del otro proyecto y súbela sin cambios.",
    ),
    "BUNDLE_TAMPERED": CommonError(
        "La receta fue modificada después de exportarse, así que no se puede confiar en ella.",
        "Vuelve a exportarla desde el proyecto de origen y súbela sin editarla.",
    ),
    "BUNDLE_TOO_LARGE": CommonError(
        "El archivo de receta es demasiado grande.",
        "Asegúrate de subir el JSON exportado por Goal Settings, no un archivo de datos.",
    ),
    "INVALID_CAPACITY": CommonError(
        "La capacidad y los casos esperados tienen que ser números mayores que cero.",
        "Escribe cuántos casos puedes atender y cuántos llegan en el mismo periodo.",
    ),
    "INVALID_COSTS": CommonError(
        "Falta indicar cuánto pesa cada tipo de error.",
        "Pon un peso mayor que cero al menos en uno: avisar de más o dejar pasar un caso.",
    ),
    "NO_OPERATING_POINTS": CommonError(
        "Este modelo se entrenó antes de que existiera esta función.",
        "Entrena una versión nueva (con el mismo plan) para poder decidir con ella.",
    ),
    "ACTION_LIST_FAILED": CommonError(
        "No pudimos crear o medir la lista de acción.",
        "Revisa que el archivo traiga el identificador de cada caso y vuelve a intentarlo.",
    ),
    "NOT_ENOUGH_CLASSES": CommonError(
        "Lo que quieres anticipar tiene menos de tres valores distintos.",
        "Con dos valores usa el modelo sí/no; si esperabas más, revisa la columna elegida.",
    ),
    "TOO_MANY_CLASSES": CommonError(
        "Lo que quieres anticipar tiene demasiados valores distintos.",
        "Agrupa los valores en menos categorías (por ejemplo, en tramos) y vuelve a intentarlo.",
    ),
    "CLASS_TOO_SMALL": CommonError(
        "Uno de los valores a anticipar tiene muy pocos casos para aprender de él.",
        "Junta ese valor con otro parecido o suma más datos antes de entrenar.",
    ),
    "NOT_ENOUGH_ROWS": CommonError(
        "Hay muy pocos casos con valor para aprender a anticipar una cantidad.",
        "Suma más datos o revisa que la columna elegida tenga números.",
    ),
    "TARGET_NOT_A_QUANTITY": CommonError(
        "Lo que quieres anticipar casi no cambia, así que no se comporta como una cantidad.",
        "Elige otra columna numérica o úsala como categoría.",
    ),
    "TIME_COLUMN_NOT_FOUND": CommonError(
        "La columna de fecha elegida para dividir ya no está en el archivo.",
        "Vuelve a elegir la división en «Validar con fechas» o usa la división al azar.",
    ),
    "TIME_COLUMN_NOT_DATES": CommonError(
        "La columna elegida para ordenar en el tiempo tiene valores que no son fechas.",
        "Elige una columna de fecha o entrena sin separar por tiempo.",
    ),
    "SOURCE_NOT_FOUND": CommonError(
        "No encontramos ese archivo en el proyecto.",
        "Recarga la página y elige un archivo de la lista.",
    ),
    "UNEXPECTED_ERROR": CommonError(
        "Algo falló inesperadamente en esta pantalla.",
        "Recarga la página; tus datos y aprobaciones guardados no se pierden. Si vuelve a pasar, "
        "comparte el código de incidente.",
    ),
    "SCORING_FAILED": CommonError(
        "No pudimos priorizar los casos de este archivo.",
        "Revisa que tenga las mismas columnas que la plantilla y vuelve a intentarlo.",
    ),
    "RETRAINING_FAILED": CommonError(
        "No pudimos crear el proyecto de reentrenamiento.",
        "Revisa el archivo reciente y la confirmación, y vuelve a intentarlo.",
    ),
    "COMPARISON_FAILED": CommonError(
        "No pudimos comparar los dos modelos con este archivo.",
        "Revisa que traiga las columnas del modelo y el resultado real, y vuelve a intentarlo.",
    ),
    "ACCEPTED_DATASET_MISSING": CommonError(
        "El archivo aceptado de este proyecto ya no está disponible en la carpeta de datos.",
        "Crea un proyecto nuevo y vuelve a subir el archivo.",
    ),
    "DATASET_NOT_ACTIVE": CommonError(
        "Este paso necesita el archivo activo del proyecto, y el que se está usando ya no lo es.",
        "Recarga la página para trabajar con el archivo activo.",
    ),
    "INVALID_INTERNAL_PATH": CommonError(
        "Un archivo interno del proyecto tiene una ubicación no válida.",
        "No lo uses; repite el paso desde el principio o crea un proyecto nuevo.",
    ),
    "SPLIT_CLASS_MISSING": CommonError(
        "No hay suficientes casos de cada resultado para separar entrenamiento, comparación y "
        "prueba final.",
        "Usa un archivo con más casos del evento o elige otro resultado a anticipar.",
    ),
    "STRATIFIED_SPLIT_FAILED": CommonError(
        "No se pudo repartir el archivo en las tres partes que necesita el entrenamiento.",
        "Usa un archivo con más filas o con más casos del evento.",
    ),
    "TRAINING_COLUMN_NOT_FOUND": CommonError(
        "Al archivo le faltan columnas que la estrategia aprobada necesita.",
        "Revisa las decisiones del proyecto o vuelve a subir el archivo correcto.",
    ),
    "EVALUATION_TARGET_MISMATCH": CommonError(
        "La parte usada para medir no tiene los dos resultados posibles.",
        "Usa un archivo con más casos del evento y vuelve a entrenar.",
    ),
    "BASELINE_MISSING": CommonError(
        "El entrenamiento no produjo el modelo simple de referencia con que se compara todo.",
        "Vuelve a generar la estrategia y entrena de nuevo.",
    ),
    "BASELINE_ORDER_INVALID": CommonError(
        "La estrategia evaluó los modelos en un orden no permitido.",
        "Vuelve a generar la estrategia desde la plantilla y entrena de nuevo.",
    ),
    "COMPOSITION_LINEAGE_MISSING": CommonError(
        "El archivo cruzado perdió el registro de qué archivos lo formaron.",
        "Vuelve a crear el cruce desde sus archivos originales.",
    ),
    "TARGET_FROM_AUXILIARY": CommonError(
        "El resultado a anticipar viene de un archivo auxiliar del cruce.",
        "Elige un resultado que esté en el archivo principal.",
    ),
    "JOIN_KEY_INCLUDED": CommonError(
        "Una columna usada para unir archivos quedó marcada para el aprendizaje.",
        "Márcala como «No usar»: identifica casos pero no explica el resultado.",
    ),
    "SOURCE_NOT_AVAILABLE": CommonError(
        "Uno de los archivos elegidos para el cruce no es un archivo aceptado de este proyecto.",
        "Recarga la página y elige archivos de la lista.",
    ),
    "SOURCE_FILE_UNAVAILABLE": CommonError(
        "No se encontró el archivo guardado de una de las fuentes del cruce.",
        "Vuelve a subir ese archivo e inténtalo de nuevo.",
    ),
    "SOURCE_HASH_MISMATCH": CommonError(
        "Uno de los archivos del cruce cambió desde que se aceptó.",
        "Vuelve a subirlo para que se revise de nuevo.",
    ),
    "SOURCE_INVALID": CommonError(
        "Uno de los archivos del cruce ya no cumple el formato aceptado.",
        "Vuelve a subirlo para revisarlo de nuevo.",
    ),
    "JOIN_KEY_NOT_FOUND": CommonError(
        "La columna elegida para unir no existe en uno de los archivos.",
        "Elige una columna que esté en ambos archivos.",
    ),
    "COMPOSITION_SPEC_INVALID": CommonError(
        "La configuración del cruce está incompleta o tiene valores no válidos.",
        "Revisa la fuente principal, las auxiliares, las columnas de unión y el responsable.",
    ),
    "TOO_MANY_SOURCES": CommonError(
        "El cruce admite como máximo cinco archivos en total.",
        "Quita alguna fuente auxiliar.",
    ),
    "SOURCE_SET_MISMATCH": CommonError(
        "Los archivos elegidos cambiaron mientras se preparaba el cruce.",
        "Recarga la página y vuelve a configurar el cruce.",
    ),
    "RESERVED_COLUMN_COLLISION": CommonError(
        "Un archivo tiene una columna con un nombre reservado para el cruce.",
        "Cambia el nombre de esa columna en tu archivo y vuelve a subirlo.",
    ),
    "OUTPUT_COLUMN_COLLISION": CommonError(
        "El cruce produciría dos columnas con el mismo nombre.",
        "Cambia el nombre de una de ellas en el archivo auxiliar.",
    ),
    "JOIN_CARDINALITY_VIOLATION": CommonError(
        "Las claves no cumplen la relación elegida entre los archivos.",
        "Revisa si cada clave aparece una sola vez o cambia la relación esperada.",
    ),
    "ROW_MULTIPLICATION_DETECTED": CommonError(
        "El cruce habría cambiado la cantidad de casos del archivo principal.",
        "Revisa que cada clave aparezca una sola vez en el archivo auxiliar.",
    ),
    "COMPOSED_FILE_TOO_LARGE": CommonError(
        "El archivo cruzado superaría el tamaño máximo permitido.",
        "Reduce filas o columnas de los archivos antes de cruzarlos.",
    ),
    "COMPOSED_DATASET_INVALID": CommonError(
        "El resultado del cruce no cumple el formato que acepta Goal Settings.",
        "Revisa los archivos de origen y vuelve a intentarlo.",
    ),
    "OUTPUT_PATH_INVALID": CommonError(
        "No se pudo guardar el archivo cruzado en la carpeta privada.",
        "Vuelve a intentarlo; si continúa, comparte el código de incidente.",
    ),
    "COMPARISON_NEEDS_OUTCOMES": CommonError(
        "Para comparar hace falta el resultado real de casos que ningún modelo haya visto.",
        "Carga casos recientes con la columna del resultado, distintos de los usados al entrenar.",
    ),
    "COMPARISON_TASK_KIND_MISMATCH": CommonError(
        "No se pueden comparar modelos que anticipan tipos de resultado distintos.",
        "Elige dos modelos de varias clases o dos modelos de cantidades.",
    ),
    "COMPARISON_METRIC_MISMATCH": CommonError(
        "Los modelos no usan la misma medida principal y no hay una comparación justa.",
        "Elige dos modelos configurados con la misma medida principal.",
    ),
    "USE_COMPARE_FOR_BINARY": CommonError(
        "Estos modelos son de sí/no y usan otra comparación.",
        "Vuelve a la comparación de modelos de sí/no.",
    ),
    "ESTIMATOR_POSITIVE_CLASS_MISMATCH": CommonError(
        "El modelo guardado no coincide con el evento que fue aprobado.",
        "No lo uses; repite el entrenamiento y la revisión humana.",
    ),
}

# Códigos del ciclo de valor y de la orquestación, agrupados por la acción que permiten.
# El código técnico se conserva en el incidente; la pantalla usa texto claro.
for _codes, _message in (
    (
        """AGENT_CHILD_PENDING AGENT_DISPATCH_MISMATCH AGENT_DISPATCH_PENDING
        AGENT_DISPATCH_SCOPE_MISMATCH AGENT_EXECUTION_FAILED AGENT_INVALID_OUTPUT
        AGENT_PROPOSAL_HASH_MISMATCH AGENT_PROVIDER_FAILURE AGENT_REFERENCE_NOT_PROVIDED
        AGENT_REPAIR_LIMIT AGENT_REPAIR_NOT_ALLOWED AGENT_REPAIR_SCOPE_MISMATCH
        AGENT_RESPONSE_HASH_REQUIRED AGENT_ROLE_NOT_ENABLED AGENT_SENSITIVE_OUTPUT
        AGENT_TURN_IN_PROGRESS AGENT_TURN_NOT_FOUND AGENT_UNSUPPORTED_CLAIM AGENT_WAITING_INPUT
        ANCESTOR_NOT_RUNNABLE BUDGET_EXHAUSTED BUDGET_INSUFFICIENT_FOR_TRAINING
        CHILD_BUDGET_OVERRIDE DELEGATION_ROLE_DENIED INVALID_AGENT_TURN_OUTCOME
        INVALID_CALLS_COST INVALID_CLOCK INVALID_CONTEXT_HASH INVALID_CONTEXT_REFRESH
        INVALID_EVENT INVALID_EVENT_TIME INVALID_INPUT INVALID_LEASE INVALID_OPERATION
        INVALID_REVIEW INVALID_ROOT_TASK INVALID_TASK_TRANSITION INVALID_TOOL_ARGUMENTS
        INVALID_TOOL_CALL INVALID_TOOL_EFFECT INVALID_TURN_LIMIT LEASE_NOT_OWNED
        OPERATION_ALREADY_FINISHED OPERATION_CONFLICT OPERATION_IN_PROGRESS
        OPERATION_LEASE_EXPIRED OPERATION_NOT_FOUND OPERATION_OUTCOME_UNKNOWN PROVIDER_DISABLED
        PROVIDER_TRANSACTION_FORBIDDEN ROOT_BUSY STALE_CONTEXT STALE_TASK_REVISION TASK_BUSY
        TASK_CHILDREN_ACTIVE TASK_NOT_ACTIVE TASK_NOT_FOUND TASK_NOT_RUNNABLE TASK_NOT_WAITING
        TASK_PARENT_REQUIRED TASK_SCOPE_MISMATCH TOOL_HUMAN_FIELD_FORBIDDEN TOOL_NOT_ALLOWED
        TOOL_ROLE_DENIED TOOL_SCOPE_MISMATCH""",
        CommonError(
            "El equipo de IA no pudo completar este paso con las condiciones actuales.",
            "Revisa la tarea y sus autorizaciones; si continúa, comparte el código de incidente.",
        ),
    ),
    (
        """AGGREGATE_COLUMN_NOT_ALLOWED CASE_ID_REQUIRED COHORT_ALREADY_SEALED COHORT_ALREADY_USED
        COHORT_HASH_MISMATCH COHORT_INCOMPLETE COHORT_INPUT_CHANGED COHORT_INTERVENTION_OVERLAP
        COHORT_LEAKAGE COHORT_NOT_FOUND COHORT_NOT_MATURE COVERAGE_REQUIRED
        DATASET_INDEX_NOT_UNIQUE DATASET_NOT_FOUND DATASET_PROFILE_MISMATCH DATA_KO
        ENTITY_ARCHIVED ENTITY_COLUMN_REQUIRED ENTITY_MISMATCH INTERVENTION_COHORT_OVERLAP
        LABEL_AVAILABILITY_NOT_DECLARED LABEL_AVAILABILITY_OUT_OF_RANGE
        LABEL_AVAILABILITY_REQUIRED LABEL_AVAILABLE_BEFORE_PREDICTION NON_FRESH_BATCH
        OUTCOME_ALREADY_OBSERVED PROVENANCE_REQUIRED RECIPE_REQUIRED REFERENCE_COHORT_REQUIRED
        SEAL_INPUT_REQUIRED SNAPSHOT_COVERAGE_INVALID SNAPSHOT_OUTCOMES_NOT_READY
        SNAPSHOT_RECIPE_NOT_AVAILABLE SNAPSHOT_RECIPE_REQUIRED SNAPSHOT_RETRAINING_INCOMPLETE
        SNAPSHOT_SOURCE_HASH_MISMATCH SOURCE_BINDING_REQUIRED SOURCE_CONTEXT_MISMATCH
        SOURCE_SCOPE_MISMATCH SPLIT_EMPTY TEMPORAL_LABEL_AVAILABILITY_REQUIRED
        UNKNOWN_COLUMN_ALIAS""",
        CommonError(
            "Los datos o su procedencia no permiten continuar con esta decisión.",
            "Revisa el archivo, su cobertura, las fechas y los permisos antes de "
            "intentarlo de nuevo.",
        ),
    ),
    (
        """ARTIFACT_HASH_NOT_SUPPORTED ARTIFACT_HASH_UNSUPPORTED ARTIFACT_NOT_FOUND
        ARTIFACT_SCOPE_MISMATCH BINDING_NOT_FOUND CANDIDATE_INCOMPLETE CANDIDATE_NOT_READY
        CANDIDATE_PROJECT_NOT_ACTIVE CANDIDATE_PROVENANCE_MISMATCH CANDIDATE_REPORT_REQUIRED
        CANDIDATE_SCOPE_MISMATCH CAPABILITY_DATA_NOT_READY CAPABILITY_EVIDENCE_CHANGED
        CAPABILITY_NOT_ACTIVE CAPABILITY_SCOPE_MISMATCH CYCLE_ALREADY_BOUND
        CYCLE_CONTEXT_MISMATCH CYCLE_NOT_ACTIVE CYCLE_SCOPE_MISMATCH CYCLE_SOURCE_REQUIRED
        MODEL_ALREADY_TRAINED MODEL_APPROVAL_NOT_CURRENT MODEL_ARTIFACT_CHANGED
        MODEL_ARTIFACT_MISSING MODEL_BINDING_REQUIRED MODEL_EVIDENCE_MISMATCH
        MODEL_NOT_RECOMMENDED PREDICTIVE_BINDING_MISMATCH PREDICTIVE_BINDING_REQUIRED
        PREDICTIVE_CONTEXT_HASH_MISMATCH PREDICTIVE_CONTEXT_INVALID PREDICTIVE_CONTEXT_MISMATCH
        PREDICTIVE_CONTEXT_NOT_CURRENT PREDICTIVE_CONTEXT_REQUIRED
        PREDICTIVE_DATASET_NOT_CURRENT PREDICTIVE_METADATA_AS_FEATURE
        PREDICTIVE_OUTCOME_MISMATCH PREDICTIVE_RECIPE_INVALID PREDICTIVE_RECIPE_MISMATCH
        PREDICTIVE_RECIPE_REQUIRED PREDICTIVE_SCOPE_MISMATCH PREDICTIVE_SOURCE_HASH_MISMATCH
        PREDICTIVE_SOURCE_NOT_ACCEPTED PREDICTIVE_SOURCE_REQUIRED PREDICTIVE_SPEC_UNCONFIRMED
        PREDICTIVE_STRATEGY_HASH_MISMATCH PREDICTIVE_STRATEGY_INVALID
        PREDICTIVE_STRATEGY_MISMATCH PREDICTIVE_STRATEGY_NOT_CURRENT PREDICTIVE_TASK_NOT_BINARY
        PREDICTIVE_WINDOW_NOT_VERIFIED PROJECT_ALREADY_BOUND RUN_MISMATCH RUN_NOT_FOUND
        SCORING_EVIDENCE_MISMATCH SCORING_EVIDENCE_REQUIRED SPEC_HASH_MISMATCH SPEC_INCOMPLETE
        SPEC_SCOPE_MISMATCH SPEC_VERSION_CONFLICT SPEC_VERSION_NOT_FOUND
        STRATEGY_CONTEXT_NOT_CURRENT STRATEGY_OPERATION_CONFLICT TRAINING_EVIDENCE_MISSING
        TRAINING_NOT_APPROVED TRAINING_OPERATION_CONFLICT TRAINING_OPERATION_IN_PROGRESS
        TRAINING_OPERATION_TERMINATED""",
        CommonError(
            "El modelo o su evidencia no están listos para este uso.",
            "Comprueba el objetivo, la versión y las revisiones requeridas antes de continuar.",
        ),
    ),
    (
        """ACTION_NOT_ALLOWED BATCH_HASH_MISMATCH BATCH_NOT_FOUND CAPACITY_REQUIRED
        COMMON_COHORT_MISMATCH COMMON_EVIDENCE_INVALID COMPARISON_STALE CURRENCY_MISMATCH
        CURRENT_POLICY_CHANGED ECONOMIC_SCENARIO_INCOMPLETE ECONOMIC_SCENARIO_NOT_CURRENT
        ECONOMIC_SCENARIO_REQUIRED EFFECT_RECORDED_CONTEXT_CHANGED EVALUATION_HASH_MISMATCH
        EVALUATION_NOT_FOUND HUMAN_REVIEW_REQUIRED HUMAN_REVIEW_RESUMED IDEMPOTENCY_CONFLICT
        MISSING_DECISION POLICIES_NOT_COMPARABLE POLICY_CHANGED POLICY_EVIDENCE_STALE
        POLICY_HASH_MISMATCH POLICY_INPUT_CHANGED POLICY_NOT_CURRENT POLICY_NOT_FOUND
        POLICY_OPERATION_CONFLICT POLICY_VERSION_CONFLICT POLICY_VERSION_STALE
        PROBABILITY_VALUATION_NOT_VERIFIED PROMOTION_NOT_SUPPORTED REPORT_CHANGED
        REPORT_MISMATCH REVIEW_MISMATCH UNKNOWN_CASE""",
        CommonError(
            "La decisión de valor no coincide con la evidencia o los límites vigentes.",
            "Revisa los supuestos, resultados y aprobaciones; confirma la versión antes de seguir.",
        ),
    ),
    (
        """CONTEXT_DATASET_MISMATCH CONTEXT_VERIFICATION_REQUIRED INVALID_AGGREGATE_OUTPUT
        INVALID_CONFIRMATION INVALID_LABEL INVALID_POLICY_ARTIFACT INVALID_POLICY_SCORES
        INVALID_PROBABILITY_BASELINE INVALID_REASON_CODE INVALID_SCENARIO_HASH
        INVALID_SCENARIO_VERSION INVALID_SELECTED_CASES LEGACY_DECISION_PLAN MISSING_STORAGE
        UNSUPPORTED_TASK""",
        CommonError(
            "No se pudo completar esta operación con el estado actual.",
            "Revisa los datos y la autorización; si continúa, comparte el código de incidente.",
        ),
    ),
):
    for _code in _codes.split():
        ERROR_MESSAGES.setdefault(_code, _message)

ERROR_MESSAGES.update(
    {
        "DATA_KO": CommonError(
            "Esta fuente no cumple las reglas de datos permitidos.",
            "Usa sólo datos sintéticos, de prueba o debidamente anonimizados y revisa su origen.",
        ),
        "TRAINING_NOT_APPROVED": CommonError(
            "El entrenamiento todavía no tiene una aprobación humana vigente.",
            "Revisa la propuesta y apruébala expresamente antes de iniciar el aprendizaje.",
        ),
        "HUMAN_REVIEW_REQUIRED": CommonError(
            "Esta decisión necesita una revisión humana vigente.",
            "Comprueba la evidencia y registra tu decisión antes de continuar.",
        ),
        "PROBABILITY_VALUATION_NOT_VERIFIED": CommonError(
            "Estos puntajes no están verificados para calcular dinero esperado.",
            "Úsalos sólo para ordenar casos y confirma por separado los supuestos económicos.",
        ),
        "PREDICTION_BATCH_FAILED": CommonError(
            "No pudimos registrar este grupo de predicciones.",
            "Revisa la versión aprobada, el período y el archivo; vuelve a intentarlo.",
        ),
        "ACTION_OBSERVATION_FAILED": CommonError(
            "No pudimos registrar la actuación o el resultado de este caso.",
            "Revisa la fecha, el caso y si ya existe un resultado registrado.",
        ),
    }
)

# Motivos por los que la IA no respondió y se usó el modo sin IA generativa.
FALLBACK_REASONS: dict[str, str] = {
    "PROVIDER_QUOTA_EXHAUSTED": "la cuenta de IA no tiene saldo disponible",
    "PROVIDER_TIMEOUT": "la IA tardó demasiado en responder",
    "PROVIDER_UNREACHABLE": "no fue posible conectarse con la IA",
    "PROVIDER_ERROR": "el servicio de IA falló de forma inesperada",
    "PROVIDER_REFUSAL": "la IA se negó a responder",
    "PROVIDER_OUTPUT_TRUNCATED": "la respuesta de la IA quedó incompleta",
    "EMPTY_PROVIDER_RESPONSE": "la IA respondió vacío",
    "INVALID_PROVIDER_RESPONSE": "la respuesta de la IA no se pudo leer",
    "INVALID_JSON": "la respuesta de la IA no tenía el formato esperado",
    "SCHEMA_INVALID": "la respuesta de la IA no cumplía el formato acordado",
    "UNKNOWN_OR_INVALID_TARGET": "la IA propuso una columna que no sirve como resultado",
    "METRIC_NOT_ALLOWED": "la IA propuso una medida fuera del catálogo permitido",
    "GUARDRAIL_MODE_NOT_ALLOWED": "la IA propuso una condición mínima no permitida",
    "UNKNOWN_COLUMN": "la IA mencionó una columna que no existe o que no se le envió",
    "FORBIDDEN_CONTENT": "la IA incluyó montos o código, que no están permitidos",
    "TECHNICAL_JARGON": "la redacción de la IA usaba términos técnicos",
    "UNVERIFIED_CONTENT": "la IA agregó cifras que no estaban en la explicación original",
    "UNVERIFIED_CLAIMS": "la respuesta incluyó afirmaciones sin evidencia suficiente",
}


def fallback_reason_text(code: str | None) -> str:
    if code is None:
        return "la IA no está activada en este equipo"
    if code.startswith("PROVIDER_HTTP_"):
        return "el servicio de IA respondió con un error"
    if code.startswith("UNVERIFIED_CLAIMS"):
        return FALLBACK_REASONS["UNVERIFIED_CLAIMS"]
    return FALLBACK_REASONS.get(code, "la respuesta de la IA no pasó los controles")


def render_database_failure(error: Exception) -> None:
    """UX-01: fallo de almacenamiento explicado, sin traza ni enlaces externos."""

    st.error(
        "No pudimos leer o guardar tus datos en este momento. "
        "No sabemos si tu última acción quedó guardada: al reintentar verás el estado real."
    )
    st.write(
        "Suele pasar cuando otra copia de Goal Settings está abierta usando la misma carpeta de "
        "datos, o cuando el disco no está disponible. Cierra la otra copia si la hay y "
        "vuelve a intentarlo."
    )
    st.button("Reintentar", type="primary", key="database-failure-retry")
    incident_id = record_incident(error, code="DATABASE_FAILURE", where="Base de datos")
    st.caption(f"Código de incidente: `{incident_id}`.")
    with st.expander("Detalle técnico"):
        st.code(f"{incident_id} · {type(error).__name__}: {error}")


def render_other_instance_warning(seen_at_label: str) -> None:
    st.warning(
        "Hay otra copia de Goal Settings usando esta misma carpeta de datos (última actividad: "
        f"{seen_at_label}). Usarlas a la vez puede dañar o bloquear tus datos. "
        "Cierra una de las dos antes de seguir."
    )


# Pantalla en que ocurre cada falla, para el registro y para la ayuda de la IA.
SCREENS: dict[str, str] = {
    "INVALID_DATASET": "Subir archivo",
    "DUPLICATE_DATASET": "Subir archivo",
    "INVALID_DATASET_COMPOSITION": "Cruzar archivos",
    "INVALID_CONTEXT": "Decisiones del proyecto",
    "INVALID_STRATEGY": "Estrategia de entrenamiento",
    "INVALID_STATE_TRANSITION": "Aprobaciones",
    "TRAINING_ALREADY_RUNNING": "Entrenamiento",
    "TRAINING_FAILED": "Entrenamiento",
    "INVALID_REPORT": "Reporte",
    "REPORT_NOT_APPROVED": "Reporte",
    "SCORING_FAILED": "Usar el modelo",
    "RETRAINING_FAILED": "Reentrenar",
    "COMPARISON_FAILED": "Comparar modelos",
    "ANALYSIS_FAILED": "Analizar tus datos",
    "PROJECT_ARCHIVE_FAILED": "Eliminar proyecto",
    "MODELING_TABLE_FAILED": "Archivos del proyecto",
    "ANALYSIS_TABLE_FAILED": "Guardar como tabla",
    "DECISION_RECORD_FAILED": "Decisiones tomadas",
    "ACTION_LIST_FAILED": "Lista de acción",
    "UNEXPECTED_ERROR": "Pantalla principal",
}


def render_error(error: BaseException, *, fallback_code: str) -> str:
    """Texto fijo, ayuda de la IA, código de incidente y detalle técnico de una falla."""

    code = str(getattr(error, "code", fallback_code) or fallback_code)
    common = _missing_feature_columns_message(error) if code == "MISSING_FEATURE_COLUMNS" else None
    common = (
        common
        or ERROR_MESSAGES.get(code)
        or ERROR_MESSAGES.get(
            fallback_code,
            CommonError(
                "No pudimos completar esta acción.",
                "Revisa los datos ingresados y vuelve a intentarlo; si continúa, comparte el "
                "código de incidente.",
            ),
        )
    )
    return render_incident(
        error,
        code=code,
        message=common.message,
        action=common.action,
        where=SCREENS.get(fallback_code, fallback_code),
    )

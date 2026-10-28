# Plan maestro — Goal Settings Analytics Studio

## 1. Propósito

Demostrar el trabajo anual de diseño de producto con IA y analítica de datos en el contexto de Goal Settings de Babel. El texto oficial del objetivo, fechas, evaluadores e indicadores aún no están documentados aquí.

## 2. Resultado esperado

Una narración trazable que conecte **objetivo → pregunta de producto → datos permitidos → análisis o experimento → decisión de diseño → evidencia → aprendizaje**. No se declararán mejoras de negocio o experiencia sin medición y aprobación reales.

## 3. Alcance

- Documentar el objetivo anual en términos verificables cuando se disponga del texto oficial.
- Seleccionar casos de diseño de producto donde IA o analítica ayuden a investigar o decidir.
- Usar la aplicación existente para explorar datos de prueba y, cuando proceda, entrenar y comparar modelos.
- Mostrar límites, supuestos, fallos y decisiones humanas.
- Preparar una demostración reproducible sin datos internos ni secretos.

Fuera de alcance hasta especificación y autorización: conexión a sistemas de Babel, ingestión de datos corporativos, evaluación automática de empleados, despliegue productivo y afirmaciones de impacto real.

## 4. Estado del software

El producto contiene Streamlit, un motor de datos tabulares, ML supervisado, reportes, IA opcional y controles de aprobación. Cada capacidad se debe verificar antes de presentarla como terminada. El flujo específico de Goal Settings no está implementado ni conectado a Babel.

## 5. Método

1. Registrar el objetivo oficial y su horizonte.
2. Formular una hipótesis y una decisión de diseño concreta.
3. Definir indicador, línea base y método de observación.
4. Identificar datos permitidos, procedencia, calidad, limitaciones y sesgos.
5. Analizar con cálculos verificables. Si se usa ML: separar train/validation/test, evitar leakage y comparar con baseline.
6. Revisar el plan antes de entrenar y los resultados antes de publicar o usar.
7. Registrar artefactos, decisiones, resultados y aprendizajes con fecha y responsable real.

## 6. Seguridad

Sólo datos sintéticos, sandbox o anonimizados con autorización. Claves fuera de Git. La IA generativa recibe únicamente contexto permitido, propone y explica; su salida se valida contra cálculos del sistema. `core/` permanece independiente de interfaz y proveedor. Ninguna aprobación se presume.

## 7. Trabajo priorizado

| ID | Trabajo | Criterio de aceptación |
|---|---|---|
| GS-00 | Preparar el repositorio local | Documentación, identidad y verificaciones registradas |
| GS-01 | Incorporar el objetivo anual oficial | Texto y criterios confirmados por la persona responsable, sin datos sensibles |
| GS-02 | Diseñar matriz de evidencia | Cada eje IA y datos ligado a una pregunta, artefacto, decisión y revisión |
| GS-03 | Crear recorrido demostrable | Demo sintética reproducible que explique IA, datos y diseño |
| GS-04 | Evaluar brechas funcionales | Backlog basado en GS-01 a GS-03, una historia por vez |

Las filas posteriores a GS-00 son propuestas, no capacidades implementadas.

## 8. Criterios de entrega

El repositorio debe abrir localmente, superar las pruebas y lint aplicables y mantener documentación coherente con el objetivo anual. La publicación en GitHub y la presentación del proyecto requieren verificación independiente.

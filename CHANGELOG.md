# Avance incremental de Goal Settings Analytics Studio

Este documento describe la evolución acumulativa del producto como parte de un
objetivo anual centrado en diseño de producto, inteligencia artificial y
analítica de datos. No funciona como una cronología: cada incremento representa
una capacidad que amplía el producto y conserva las capacidades anteriores.

El proyecto utiliza información sintética o autorizada. No contiene resultados
empresariales reales ni supone una integración implementada con Babel.

## Objetivo de producto

Conectar de forma trazable:

`objetivo → pregunta de diseño → datos permitidos → análisis o experimento → decisión → evidencia → aprendizaje`.

El producto busca que una persona pueda:

- formular un objetivo de Goal Settings y convertirlo en preguntas verificables;
- comprender la calidad y el alcance de sus datos;
- explorar alternativas mediante analítica y ML tradicional;
- utilizar IA como apoyo para interpretar, nunca para inventar métricas;
- aprobar las decisiones relevantes antes de activar un modelo;
- conservar evidencia de lo aprendido y del valor para diseño de producto.

## Principios acumulativos

- Los incrementos se expresan como capacidades del producto, no como cortes temporales.
- Cada capacidad debe aportar valor al objetivo y contar con validación verificable.
- El ML tradicional se entrena y evalúa con separación de datos y comparación contra baseline.
- La IA propone, explica y orienta; no ejecuta código arbitrario ni reemplaza el análisis verificable.
- Las decisiones sensibles requieren confirmación humana vigente.
- Los ejemplos, escenarios y resultados incluidos son sintéticos.
- La trazabilidad registra estados, decisiones y evidencias sin almacenar valores sensibles.

## Incremento 1 — Definición del objetivo

**Propósito:** convertir un objetivo anual en una unidad de trabajo clara para producto.

Capacidades acumuladas:

- creación de proyectos con identidad, propósito y alcance;
- definición del problema, población, decisión y resultado esperado;
- registro de indicadores sin presentarlos como métricas oficiales de Babel;
- contexto de negocio versionado y reutilizable;
- plantillas para formular objetivos y preguntas de diseño;
- huella verificable del contexto para detectar cambios relevantes;
- invalidación de aprobaciones cuando cambia la definición del objetivo.

Identificadores relacionados: `CTX-01`, `STRAT-01`, `V2-01…V2-04`.

## Incremento 2 — Admisión y comprensión de datos

**Propósito:** conocer qué información existe, qué calidad tiene y para qué puede utilizarse.

Capacidades acumuladas:

- admisión de CSV con límites de tamaño, formato y estructura;
- almacenamiento aislado por proyecto;
- cuarentena de archivos que no cumplen los contratos;
- perfilado determinista de columnas, tipos, nulos y cardinalidad;
- asignación de roles para objetivo, identificadores y variables de entrada;
- exclusión explícita de campos sensibles o no autorizados;
- relaciones entre archivos y composición declarativa;
- carga individual y en lote;
- registro de procedencia y huellas de contenido;
- escenarios sintéticos para recorridos, clasificación, cantidades y variables temporales.

Identificadores relacionados: `DATA-01`, `DATA-02.1…DATA-02.3`,
`GA-02a…GA-06b`, `V2-08…V2-11`.

## Incremento 3 — Calidad y preparación analítica

**Propósito:** transformar datos permitidos en insumos consistentes para análisis y experimentos.

Capacidades acumuladas:

- reglas de calidad ampliables y resultados explicables;
- análisis automáticos declarativos;
- combinaciones, derivaciones y cálculos reproducibles;
- preparación de tablas por caso sin fuga de información;
- recetas editables y reutilizables;
- conservación de la receta junto con su tabla resultante;
- cohortes y recorridos construidos desde actividad sintética;
- embudos y tiempos entre etapas;
- consultas de solo lectura sobre resultados agregados;
- exportación e importación de recetas y decisiones.

Identificadores relacionados: `GA-00…GA-06`, `GA-10…GA-14`,
`C2-03`, `C2-04`, `V2-18…V2-24c`.

## Incremento 4 — Estrategia y experimento verificable

**Propósito:** convertir la pregunta del objetivo en una estrategia analítica reproducible.

Capacidades acumuladas:

- catálogo de plantillas para clasificación, regresión y análisis temporal;
- selección explícita de objetivo, variables y reglas de exclusión;
- fingerprint de estrategia y datos;
- aprobación humana previa al entrenamiento;
- invalidación automática cuando cambian datos, contexto o estrategia;
- separación reproducible de train, validation y test;
- preprocessing aprendido exclusivamente desde train;
- comparación de candidatas contra un baseline;
- selección antes de utilizar test;
- estados honestos para experimentos incompletos o no comparables.

Identificadores relacionados: `STRAT-01`, `ML-01`, `ML-02`,
`GA-07a…GA-08`.

## Incremento 5 — Decisión asistida por modelos

**Propósito:** usar modelos aprobados para apoyar una decisión sin ocultar sus límites.

Capacidades acumuladas:

- predicción de clase o cantidad según el tipo de modelo;
- probabilidades y rangos presentados con contexto;
- explicación de por qué un caso recibe una sugerencia;
- listas de acción separadas de los resultados observados;
- reglas para evitar confundir una puntuación con una certeza;
- comparación entre modelo vigente y modelo candidato;
- reemplazo condicionado a evidencia y confirmación humana;
- trazabilidad de la versión exacta utilizada en cada decisión;
- medición posterior para saber si la acción produjo el efecto esperado.

Identificadores relacionados: `V2-12…V2-17`, `V2-25…V2-26`,
`C2-02`, `C2-06…C2-09`.

## Incremento 6 — Evidencia, resultados y aprendizaje

**Propósito:** vincular acciones con resultados observables y convertirlos en aprendizaje.

Capacidades acumuladas:

- registro separado de actuación y resultado;
- cobertura y maduración de resultados;
- exclusión de resultados aún no observables;
- métricas agregadas sin exponer identificadores de casos;
- comparación común entre alternativas;
- estimaciones económicas declaradas como supuestos;
- seguimiento de ingresos, ahorro o impacto solamente cuando existen datos válidos;
- resultados simulados claramente identificados en la demo;
- ciclo de aprendizaje que conserva receta, decisión, resultado y conclusión.

Identificadores relacionados: `MV-01…MV-06`, `MV-11…MV-14`,
`V2-17`, `V2-25b`.

## Incremento 7 — Supervisión y salud del producto analítico

**Propósito:** mantener visibles el estado, la vigencia y los límites de cada modelo.

Capacidades acumuladas:

- estado de salud para modelos de distintos tipos;
- comparación entre baseline, modelo activo y candidatas;
- detección de cambios en cobertura y distribución;
- reentrenamiento mediante receta completa;
- conservación del modelo activo hasta que exista una alternativa válida;
- recuperación ante operaciones incompletas;
- indicadores del ciclo del proyecto;
- activación condicionada a evidencias y aprobaciones vigentes.

Identificadores relacionados: `MON-01…MON-04`, `MV-06`,
`MV-12…MV-13`.

## Incremento 8 — IA como guía del objetivo

**Propósito:** ayudar a comprender el objetivo, los datos y los resultados con lenguaje claro.

Capacidades acumuladas:

- proveedor de IA intercambiable y configuración local opcional;
- funcionamiento determinista cuando no hay proveedor disponible;
- herramientas limitadas y de solo lectura para asistentes;
- autorización explícita de campos antes de enviar contexto;
- interpretación basada en cifras calculadas por el sistema;
- propuestas de análisis convertidas en borradores, nunca ejecutadas automáticamente;
- preguntas sobre datos con respuestas respaldadas por resultados verificables;
- sugerencias sobre qué anticipar a partir del objetivo y del recorrido;
- manejo claro de respuestas incompletas, límites y errores del proveedor;
- declaración visible de cuándo intervino IA generativa.

Identificadores relacionados: `AI-01`, `GA-04a…GA-04c`,
`GA-09`, `V2-05…V2-07`, `V2-24d…V2-24e`, `RU-15`.

## Incremento 9 — Experiencia guiada de producto

**Propósito:** permitir que una persona avance desde el objetivo hasta la evidencia sin necesitar
conocer la arquitectura interna.

Capacidades acumuladas:

- navegación por proyectos y pasos principales;
- guía contextual que indica estado, siguiente acción y motivo;
- lenguaje común para datos, modelos y decisiones;
- modos simple y avanzado;
- aceptación explícita de recomendaciones de columnas;
- mensajes de error con salida concreta;
- exploración de datos integrada al recorrido;
- continuidad entre sesiones;
- accesibilidad visual, contraste y estructura semántica;
- recorridos sintéticos que cubren distintas categorías y tipos de objetivo.

Identificadores relacionados: `UX-01`, `RU-01…RU-15`,
`V2-01…V2-12`, `V2-27…V2-33b`.

## Incremento 10 — Coordinación de especialistas

**Propósito:** dividir el trabajo asistido por IA en responsabilidades acotadas y comprobables.

Capacidades acumuladas:

- coordinación del objetivo y del contexto de negocio;
- especialista de datos y maduración;
- especialista de modelado y preparación del experimento;
- especialista de valor y política de decisión;
- supervisor de estado y recuperación;
- límites de herramientas por especialidad;
- resultados estructurados y verificables;
- pausa humana antes de entrenamiento y activación;
- prohibición de crear políticas nuevas de forma implícita.

Identificadores relacionados: `MV-07…MV-10`.

## Incremento 11 — Trazabilidad y operación segura

**Propósito:** conservar la relación entre código, configuración, datos permitidos y capacidad activa.

Capacidades acumuladas:

- persistencia transaccional en SQLite para proyectos locales;
- eventos de dominio sin datos sensibles;
- operaciones idempotentes y recuperación de fallas;
- backups y restauración del espacio local;
- contenedor reproducible para la aplicación;
- contratos de configuración y validación de dependencias;
- comprobaciones automáticas de formato, tipos, seguridad y pruebas;
- evidencia firmada para vincular un artefacto consumidor con una versión exacta;
- claves privadas excluidas del repositorio;
- sandbox público separado de datos y credenciales reales.

Identificadores relacionados: `HITL-01`, `QA-01`, `SDLC-01`,
`SDLC-02.1…SDLC-02.2`.

## Incremento 12 — Reporte del objetivo

**Propósito:** presentar el aprendizaje del proyecto de forma verificable y útil para Goal Settings.

Capacidades acumuladas:

- reporte generado desde datos y artefactos del proyecto;
- resumen del objetivo, estrategia, experimento y resultado;
- métricas vinculadas al modelo y dataset exactos;
- aprobación humana antes de activar resultados;
- exportación reproducible;
- diferenciación entre resultado real, simulación y supuesto;
- sección de aprendizajes para diseño de producto;
- próximos pasos expresados como nuevos incrementos del objetivo.

Identificadores relacionados: `REPORT-01`, `MV-13…MV-14`.

## Estado acumulado del producto

Goal Settings Analytics Studio dispone de un recorrido local que integra:

1. definición del objetivo y contexto;
2. admisión y comprensión de datos;
3. preparación de una estrategia;
4. entrenamiento y comparación de ML tradicional;
5. aprobaciones humanas;
6. interpretación opcional mediante IA;
7. seguimiento de decisiones y resultados;
8. generación de evidencia y aprendizaje.

La demo demuestra capacidades con datos sintéticos. No acredita cumplimiento del objetivo anual,
impacto empresarial ni aprobación de Babel.

## Próximos incrementos del objetivo

- incorporar la definición oficial del objetivo anual y sus indicadores autorizados;
- acordar qué evidencia necesita Babel y cómo debe presentarse;
- seleccionar un caso de diseño de producto con datos permitidos;
- establecer una métrica base y un criterio de avance;
- ejecutar un experimento reproducible;
- documentar el aprendizaje y su aplicación al proceso de diseño;
- decidir si corresponde integrar el producto con alguna fuente o sistema corporativo.

Cada nuevo avance debe ampliar una capacidad de este documento o agregar un incremento claramente
vinculado al objetivo, sin convertir el changelog en una bitácora temporal.

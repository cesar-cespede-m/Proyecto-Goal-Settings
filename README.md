# Proyecto-Goal-Settings

**Goal Settings Analytics Studio** reúne una aplicación local y la evidencia de un objetivo anual de diseño de producto con dos ejes: **inteligencia artificial** y **analítica de datos**. El contexto de evaluación es Babel. El proyecto documenta el proceso de Goal Settings sin atribuirle integraciones o resultados empresariales no verificados.

## Propósito

Conectar **objetivo → pregunta de diseño → datos permitidos → análisis o experimento → decisión → evidencia → aprendizaje**. La definición oficial del objetivo anual, sus indicadores, fechas y criterios de evaluación debe incorporarse desde Babel; no está contenida en este repositorio.

## Estado del producto

| Área | Capacidad disponible | Trabajo específico de Goal Settings |
|---|---|---|
| Datos | CSV, perfilado, calidad, exploración y composición | Vincular análisis con el objetivo y su evidencia |
| ML | Modelos supervisados, baseline, evaluación y aprobaciones | Elegir experimentos útiles para una pregunta de diseño |
| IA | Guía y asistentes opcionales con herramientas limitadas | Explicar el aprendizaje de IA y diseño con evidencia |
| Seguimiento | Decisiones, acciones y resultados en escenarios de prueba | Definir indicadores oficiales y revisar avance |
| Babel | No hay integración verificada | Pendiente de especificar |

Los ejemplos son sintéticos. Una demo no acredita impacto real, cumplimiento del objetivo ni aprobación de Babel.

## Ejecutar localmente

Requiere Python 3.12, 3.13 o 3.14. En Windows, abre [INICIAR_GOAL_SETTINGS.cmd](INICIAR_GOAL_SETTINGS.cmd) para un espacio vacío o [INICIAR_DEMO.cmd](INICIAR_DEMO.cmd) para una demo sintética. También puedes ejecutar:

```powershell
py -3.14 scripts/local_launcher.py --install-only
py -3.14 scripts/local_launcher.py --check
py -3.14 scripts/local_launcher.py
py -3.14 scripts/local_launcher.py --demo
```

La aplicación abre en <http://127.0.0.1:8501>. El lanzador instala dependencias en `.venv` y guarda el trabajo en `runtime/`; ambas rutas se excluyen de Git. La primera instalación requiere conexión. La demo incluye revisiones simuladas, identificadas como tales.

Para IA opcional, copia `.env.example` a `.env`, configura un proveedor y una clave local, y usa [INICIAR_CON_IA.cmd](INICIAR_CON_IA.cmd). Sin IA generativa funcionan los cálculos deterministas y reportes por plantilla. No publiques claves, bases de datos, capturas sensibles ni información interna de Babel.

Con Docker Compose:

```sh
docker compose up -d --build
docker compose ps
```

Abre <http://127.0.0.1:8501> cuando el servicio esté listo. No ejecutes Docker y Streamlit nativo sobre la misma carpeta `runtime/` simultáneamente.

## Documentación

- [Plan maestro](PLAN_MAESTRO_GOAL_SETTINGS.md).
- [Objetivos y evidencia](docs/OBJETIVOS_Y_EVIDENCIA.md).
- [Datos, IA y ML](docs/DATOS_IA_ML.md).
- [Arquitectura y operación](docs/ARQUITECTURA_Y_OPERACION.md).
- [Estado y próximos pasos](docs/ESTADO_Y_ROADMAP.md).
- [Revisión local previa a publicación](docs/REVISION_LOCAL_2026-10-28.md).

Los cambios en identificadores internos, contratos o rutas persistidas requieren una migración validada antes de reutilizar datos de otra instalación.

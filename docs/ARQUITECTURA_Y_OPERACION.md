# Arquitectura y operación

## Componentes

- `app/`: interfaz Streamlit y experiencia guiada.
- `core/data_engine/` y `core/analysis/`: carga, calidad y análisis tabular.
- `core/ml_engine/`: entrenamiento y evaluación supervisada.
- `core/domain/` y `core/contracts/`: decisiones, aprobaciones y contratos.
- `core/ai_engine/`: proveedor opcional, propuestas y validación.
- `core/db/`: persistencia local SQLite.
- `scripts/`: demos sintéticas, utilidades y ejecución local.

`core/` no debe depender de Streamlit ni de un proveedor LLM. Los cambios en contratos persistidos, variables o rutas internas necesitan pruebas de compatibilidad antes de importar datos existentes.

## Operación local

Desde la raíz del repositorio:

```powershell
py -3.14 scripts/local_launcher.py --install-only
py -3.14 scripts/local_launcher.py --check
py -3.14 scripts/local_launcher.py
```

La base y los artefactos locales se guardan en `runtime/`. Mantener esa carpeta fuera de Git, no mezclar ejecución Docker y nativa sobre la misma base, y respaldarla antes de migrar o actualizar datos.

Para comprobar cambios:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check .
```

No declarar una prueba aprobada hasta ejecutarla y registrar su resultado real.

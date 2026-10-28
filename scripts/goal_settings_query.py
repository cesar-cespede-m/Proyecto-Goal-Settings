"""Consulta de sólo lectura sobre modelos y datos aprobados (ciclo 2, C2-03).

Para que otro sistema use Goal Settings sin abrir la interfaz y sin saltarse ninguna aprobación:

    python scripts/goal_settings_query.py predict --run RUN_ID --file casos.csv --actor sistema-x
    python scripts/goal_settings_query.py ask --project PROJECT_ID --question "¿...?"

`predict` sólo funciona con un modelo cuyo reporte una persona aprobó y que sigue vigente; el
uso queda auditado igual que en la pantalla. `ask` arma el cálculo con IA (si está activa) a
partir de los tipos de columna, lo ejecuta con el motor y devuelve el resultado; no envía filas
a la IA. Ninguno de los dos cambia decisiones, modelos ni archivos.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.ai_engine.data_questions import DataQuestionService  # noqa: E402
from core.analysis.engine import AnalysisEngine, load_project_source  # noqa: E402
from core.analysis.guided import describe_plan  # noqa: E402
from core.data_engine import analysis_schema  # noqa: E402
from core.db import get_connection, migrate  # noqa: E402
from core.db.repositories import WorkflowRepository  # noqa: E402
from core.ml_engine.scoring import ScoringService  # noqa: E402


def _runtime() -> Path:
    return Path(os.environ.get("GOAL_SETTINGS_RUNTIME_DIR", ROOT / "runtime"))


def _provider():
    from core.ai_engine import LLMConfigError, LLMSettings, build_provider

    try:
        env_file = Path(os.environ.get("GOAL_SETTINGS_ENV_FILE", ROOT / ".env"))
        return build_provider(LLMSettings.from_env(env_file=env_file))
    except LLMConfigError:  # sin IA la consulta sigue disponible con lo que no requiere IA
        return None


def predict(repository: WorkflowRepository, runtime: Path, args: argparse.Namespace) -> dict:
    service = ScoringService(repository, runtime)
    _, _, proposal = service.approved_run(args.run)
    content = Path(args.file).read_bytes()
    if proposal.task_type == "binary_classification":
        result = service.score(
            args.run, filename=Path(args.file).name, content=content, actor_label=args.actor
        )
        cases = [
            {"row": case.row_number, "id": case.case_id, "score": case.score, "alert": case.alert}
            for case in result.cases
        ]
    else:
        result = service.predict(
            args.run, filename=Path(args.file).name, content=content, actor_label=args.actor
        )
        cases = [
            {"row": case.row_number, "id": case.case_id, "prediction": case.prediction}
            for case in result.cases
        ]
    return {"run_id": args.run, "task_type": proposal.task_type, "cases": cases}


def ask(repository: WorkflowRepository, runtime: Path, args: argparse.Namespace) -> dict:
    datasets = repository.list_datasets(args.project, admission_status="ACCEPTED")
    if not datasets:
        raise SystemExit("El proyecto no tiene archivos aceptados")
    names = {Path(str(row["original_filename_display"])).stem: row for row in datasets[:5]}
    hidden = {"personal", "identifier"}
    schemas = {
        name: {
            column: kind
            for column, kind in analysis_schema(json.loads(row["profile_json"])).items()
            if kind not in hidden
        }
        for name, row in names.items()
    }
    answer = DataQuestionService(_provider()).plan(args.question, schemas)
    if answer.plan is None:
        return {"answered": False, "mode": answer.mode, "reason": answer.reason}
    if args.actor:
        repository.record_ai_data_shared(
            project_id=args.project,
            purpose="query_cli",
            columns=[
                f"{source}.{column}" for source, schema in schemas.items() for column in schema
            ],
            rows=0,
        )
    sources = {
        name: load_project_source(repository, runtime, str(names[name]["id"]))
        for name in answer.plan.sources
    }
    result = AnalysisEngine().run(answer.plan, sources)
    return {
        "answered": True,
        "how": describe_plan(answer.plan),
        "columns": [column.name for column in result.columns],
        "rows": result.rows,
        "total_rows": result.total_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("predict", help="Usar un modelo aprobado y vigente")
    one.add_argument("--run", required=True)
    one.add_argument("--file", required=True)
    one.add_argument("--actor", required=True, help="Quién usa el modelo (queda auditado)")
    two = commands.add_parser("ask", help="Preguntar a los datos de un proyecto")
    two.add_argument("--project", required=True)
    two.add_argument("--question", required=True)
    two.add_argument("--actor", default="", help="Quién consulta (audita lo que va a la IA)")
    args = parser.parse_args(argv)

    runtime = _runtime()
    connection = get_connection(runtime / "goal_settings.sqlite3")
    migrate(connection)
    try:
        repository = WorkflowRepository(connection)
        output = (predict if args.command == "predict" else ask)(repository, runtime, args)
    finally:
        connection.close()
    print(json.dumps(output, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

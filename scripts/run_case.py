"""Ejecuta un caso completo desde terminal, con aprobación humana explícita en dos pasos.

Paso 1 (prepara, no entrena):
    python scripts/run_case.py prepare --csv data/samples/ai4i2020.csv \
        --decisions data/metadata/ai4i_context.json --project "Caso A - AI4I"

Paso 2 (la persona revisa la estrategia impresa y aprueba su fingerprint exacto):
    python scripts/run_case.py approve-and-run --strategy-id <id> \
        --fingerprint <fingerprint> --actor "Nombre Apellido" --comment "Revisado"

El paso 2 falla si el fingerprint no coincide con la estrategia vigente: nadie aprueba algo
que no vio. La base y los artefactos quedan en --runtime (por defecto runtime/, fuera de Git).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.data_engine import DataIngestionService  # noqa: E402
from core.db import get_connection, migrate  # noqa: E402
from core.db.repositories import WorkflowRepository  # noqa: E402
from core.domain.context_service import BusinessContextService  # noqa: E402
from core.domain.strategy_service import StrategyService  # noqa: E402
from core.ml_engine import MLRunService  # noqa: E402

_PENDING_ACTORS = {"", "pendiente", "pending", "todo"}


def _repository(runtime: Path) -> WorkflowRepository:
    runtime.mkdir(parents=True, exist_ok=True)
    connection = get_connection(runtime / "goal_settings.sqlite3", use_wal=True)
    migrate(connection)
    return WorkflowRepository(connection)


def _require_actor(label: str, field: str) -> None:
    if label.strip().casefold() in _PENDING_ACTORS:
        raise SystemExit(f"{field} está PENDIENTE: una persona responsable debe declararse.")


def prepare(args: argparse.Namespace) -> None:
    decisions = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
    _require_actor(decisions["context"]["decision_actor_label"], "decision_actor_label")
    runtime = Path(args.runtime)
    repository = _repository(runtime)
    project_id = repository.create_project(args.project)
    admission = DataIngestionService(repository, runtime).ingest_bytes(
        project_id=project_id,
        request=decisions["dataset"],
        content=Path(args.csv).read_bytes(),
    )
    context = BusinessContextService(repository, runtime).create_version(
        project_id=project_id,
        dataset_id=admission.metadata.dataset_id,
        context=decisions["context"],
    )
    strategy = StrategyService(repository).create_template_version(
        project_id=project_id,
        context_version_id=context.context_version_id,
    )
    print(json.dumps(strategy.proposal.model_dump(mode="json"), indent=2, ensure_ascii=False))
    print(
        "\nRevise la estrategia. Para aprobarla y entrenar:\n"
        f"  python scripts/run_case.py --runtime {args.runtime} approve-and-run "
        f"--strategy-id {strategy.strategy_version_id} "
        f'--fingerprint {strategy.fingerprint} --actor "<su nombre>"'
    )


def approve_and_run(args: argparse.Namespace) -> None:
    _require_actor(args.actor, "--actor")
    runtime = Path(args.runtime)
    repository = _repository(runtime)
    strategy = repository.get_strategy(args.strategy_id)
    if strategy["fingerprint"] != args.fingerprint:
        raise SystemExit("El fingerprint no coincide con la estrategia: no se aprueba nada.")
    repository.decide_training(
        strategy_version_id=args.strategy_id,
        decision="APPROVED",
        actor_label=args.actor,
        comment=args.comment,
    )
    result = MLRunService(repository, runtime).execute(args.strategy_id)
    output = result.model_dump(mode="json")
    destination = runtime / "reports" / f"run-{result.run_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResultado guardado en {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime", default=str(ROOT / "runtime"))
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--csv", required=True)
    prepare_parser.add_argument("--decisions", required=True)
    prepare_parser.add_argument("--project", required=True)
    prepare_parser.set_defaults(handler=prepare)

    run_parser = commands.add_parser("approve-and-run")
    run_parser.add_argument("--strategy-id", required=True)
    run_parser.add_argument("--fingerprint", required=True)
    run_parser.add_argument("--actor", required=True)
    run_parser.add_argument("--comment")
    run_parser.set_defaults(handler=approve_and_run)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

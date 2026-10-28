"""Smoke QA-01.6 del asistente de contexto con el proveedor configurado.

Usa un CSV sintético generado en memoria y una base temporal; nunca datos reales.

    python scripts/smoke_llm_assistant.py            # sólo muestra lo que se enviaría
    python scripts/smoke_llm_assistant.py --live     # llamada real (facturable) según .env

Imprime modo, motivo de fallback y traza técnica (proveedor, modelo, response_id, latencia,
tokens), nunca la clave. Sale con código 1 si una comprobación de minimización falla.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.ai_engine import LLMSettings, build_provider  # noqa: E402
from core.ai_engine.context_assistant import (  # noqa: E402
    ContextAssistant,
    apply_turn,
    build_assistant_request,
)
from core.ai_engine.outgoing_review import review_outgoing  # noqa: E402
from core.data_engine import DataIngestionService  # noqa: E402
from core.data_engine.models import DatasetProfileV1  # noqa: E402
from core.db import get_connection, migrate  # noqa: E402
from core.db.repositories import WorkflowRepository  # noqa: E402
from core.domain.assistant import SafeProfileForLLMV1  # noqa: E402
from core.domain.draft_service import ContextDraftService  # noqa: E402

MESSAGE = (
    "El archivo tiene lecturas sintéticas de sensores de máquinas de una planta de prueba. "
    "Quiero anticipar qué máquinas van a fallar para decidir cuáles revisar al inicio del turno."
)
MARKER_CLASS = "FALLA-MARCADOR-NO-DEBE-SALIR"


def _synthetic_csv(rows: int = 300) -> bytes:
    rng = np.random.default_rng(20261019)
    failure = rng.choice([MARKER_CLASS, "sin_falla"], size=rows, p=[0.2, 0.8])
    torque = np.where(failure == MARKER_CLASS, rng.normal(60, 5, rows), rng.normal(40, 5, rows))
    frame = pd.DataFrame(
        {
            "machine_id": [f"M-{index:05d}" for index in range(rows)],
            "air_temperature": rng.normal(300, 2, rows).round(2),
            "torque": torque.round(2),
            "tool_wear_after_inspection": rng.integers(0, 250, rows),
            "failure": failure,
        }
    )
    return frame.to_csv(index=False).encode("utf-8")


def _check(condition: bool, label: str, failures: list[str]) -> None:
    print(f"  [{'OK' if condition else 'FALLA'}] {label}")
    if not condition:
        failures.append(label)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Realiza la llamada real al proveedor")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    settings = LLMSettings.from_env(env_file=ROOT / ".env")
    provider = build_provider(settings) if args.live else None
    print(
        f"LLM habilitado: {settings.active} · proveedor: {settings.provider} · "
        f"modelo: {settings.model or '-'} · llamada real: {provider is not None}"
    )

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="goal_settings-smoke-") as tmp:
        root = Path(tmp)
        connection = get_connection(root / "smoke.sqlite3")
        migrate(connection)
        repository = WorkflowRepository(connection)
        project_id = repository.create_project("Smoke QA-01.6")
        admission = DataIngestionService(repository, root / "private").ingest_bytes(
            project_id=project_id,
            request={
                "original_filename_display": "smoke-sintetico.csv",
                "source_name": "Generado por scripts/smoke_llm_assistant.py",
                "classification": "synthetic",
                "authorization_status": "approved",
            },
            content=_synthetic_csv(),
        )
        dataset = repository.get_dataset(admission.metadata.dataset_id)
        profile = DatasetProfileV1.model_validate(json.loads(dataset["profile_json"]))
        stored = ContextDraftService(repository, root / "private").open(
            project_id, admission.metadata.dataset_id
        )
        review = review_outgoing(MESSAGE, SafeProfileForLLMV1.from_profile(profile))
        request = build_assistant_request(review, stored.draft, profile)

        print("\nComprobaciones de minimización del envío:")
        sent = request.system + request.user
        _check(MARKER_CLASS not in sent, "el valor del evento no sale del equipo", failures)
        _check(dataset["sha256"] not in sent, "el hash del archivo no sale", failures)
        _check("smoke-sintetico" not in sent, "el nombre del archivo no sale", failures)
        _check("M-00001" not in sent, "ningún valor de fila sale", failures)
        print(f"  Tamaño del envío: {len(request.user)} caracteres; hallazgos: {review.findings}")

        result = ContextAssistant(provider).next_turn(review, stored.draft, profile)
        print(f"\nAsistente → modo: {result.mode} · motivo: {result.reason}")
        print("  Traza:", json.dumps(result.trace, ensure_ascii=False))
        print("  Mensaje:", result.turn.assistant_message)
        for question in result.turn.questions:
            print(f"  Pregunta [{question.field.value}]: {question.prompt}")
        for proposal in result.turn.proposals:
            print(f"  Propuesta [{proposal.field}]: {proposal.value} — {proposal.explanation}")
        for note in result.turn.feature_notes:
            print(
                f"  Nota {note.column_ref}: post-evento={note.possible_post_event} — "
                f"{note.explanation}"
            )
        apply_turn(stored.draft, result.turn, profile)
        _check(
            not any(
                stored.draft.get(field).status.value == "CONFIRMED" for field in stored.draft.fields
            ),
            "la IA no confirmó ningún campo",
            failures,
        )
        if args.live:
            _check(result.mode == "LLM", "respuesta real del proveedor aceptada", failures)
        connection.close()

    print("\nResultado:", "PASS" if not failures else f"FALLA ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

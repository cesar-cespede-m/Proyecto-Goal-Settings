"""Mide qué hace cada actor (IA generativa, ML, persona, reglas) y el resultado del flujo.

Recorre los tres datasets sintéticos de `data/synthetic/` en una base temporal:

1. IA generativa (asistente): un turno con una descripción en lenguaje común; se compara su
   propuesta con la verdad conocida de cada dataset (resultado, datos posteriores al evento,
   identificadores y constantes).
2. Persona: se simula con las decisiones correctas documentadas en `data/synthetic/README.md`
   y se cuentan las confirmaciones que exige el producto.
3. ML: entrena y evalúa con la estrategia fija; se compara contra el modelo de referencia y
   contra alertar al azar (prevalencia).
4. IA generativa (explicación y reporte): se registra si su redacción pasó los controles.

    python scripts/measure_outcomes.py            # sin IA generativa (modo plantilla)
    python scripts/measure_outcomes.py --live     # con el proveedor de .env (facturable)
    python scripts/measure_outcomes.py --live --json salida.json

Las cifras son exploratorias sobre datos sintéticos; no son resultados de negocio.
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.ai_engine import LLMSettings, NarrativeService, build_provider  # noqa: E402
from core.ai_engine.context_assistant import ContextAssistant, apply_turn  # noqa: E402
from core.ai_engine.outgoing_review import review_outgoing  # noqa: E402
from core.ai_engine.strategy_explainer import StrategyExplainer  # noqa: E402
from core.data_engine import DataIngestionService  # noqa: E402
from core.data_engine.models import DatasetProfileV1  # noqa: E402
from core.db import get_connection, migrate  # noqa: E402
from core.db.repositories import WorkflowRepository  # noqa: E402
from core.domain.assistant import (  # noqa: E402
    Availability,
    DraftField,
    DraftUse,
    FieldSource,
    SafeProfileForLLMV1,
    column_refs,
)
from core.domain.draft_service import ContextDraftService  # noqa: E402
from core.domain.strategy import StrategyProposalV1  # noqa: E402
from core.domain.strategy_service import StrategyService  # noqa: E402
from core.ml_engine import MLRunService  # noqa: E402
from core.reporting import ReportService  # noqa: E402

ACTOR = "medicion-automatica (persona simulada)"

CASES: dict[str, dict[str, Any]] = {
    "mantenimiento_maquinas.csv": {
        "message": (
            "Tengo lecturas de sensores de máquinas de una planta. Quiero anticipar qué máquinas "
            "van a fallar para decidir cuáles revisar al inicio del turno."
        ),
        "target": "falla_maquina",
        "post_event": {"horas_detenida_post_falla"},
        "moment": "Al inicio del turno, antes de revisar las máquinas",
        "metric": "recall",
    },
    "abandono_suscripcion.csv": {
        "message": (
            "Son datos de suscriptores de un servicio. Quiero saber a quiénes contactar este mes "
            "antes de que se den de baja."
        ),
        "target": "abandona",
        "post_event": {"encuesta_salida_completada"},
        "moment": "A inicio de mes, antes de cualquier solicitud de baja",
        "metric": "average_precision",
    },
    "retraso_entregas.csv": {
        "message": (
            "Son envíos de una empresa de reparto. Quiero anticipar qué envíos llegarán tarde "
            "para reforzarlos antes de despacharlos."
        ),
        "target": "entregado_tarde",
        "post_event": {"minutos_retraso_real", "reclamo_cliente"},
        "moment": "Al planificar el despacho, antes de que el envío salga",
        "metric": "f1",
    },
    "reservas_restaurante.csv": {
        "message": (
            "Son reservas de una cadena de restaurantes. Quiero anticipar qué reservas no van a "
            "llegar para decidir a cuáles llamar y liberar mesas a tiempo."
        ),
        "target": "no_asistio",
        "post_event": {"propina_registrada", "minutos_espera_mesa"},
        "moment": "La mañana del día de la reserva, antes del servicio",
        "metric": "average_precision",
    },
}


def _ingest(repository: WorkflowRepository, storage: Path, name: str) -> tuple[str, str]:
    project_id = repository.create_project(f"Medición · {name}")
    admission = DataIngestionService(repository, storage).ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": name,
            "source_name": "Generado por scripts/generate_synthetic_datasets.py (datos falsos)",
            "classification": "synthetic",
            "authorization_status": "approved",
        },
        content=(ROOT / "data" / "synthetic" / name).read_bytes(),
    )
    return project_id, admission.metadata.dataset_id


def _assistant_quality(draft, case: dict[str, Any], profile: DatasetProfileV1) -> dict[str, Any]:
    traps_id = {c.name for c in profile.columns if c.possible_identifier or c.constant}
    flagged = {f.column for f in draft.features if f.possible_post_event}
    suggested_exclude = {
        f.column
        for f in draft.features
        if f.note_source is FieldSource.LLM and f.use is DraftUse.EXCLUDE
    }
    leaks = case["post_event"]
    target = draft.get(DraftField.TARGET_COLUMN)
    proposed = [name.value for name in DraftField if draft.get(name).source is FieldSource.LLM]
    return {
        "target_correct": target.source is FieldSource.LLM and target.value == case["target"],
        "post_event_found": sorted(leaks & flagged),
        "post_event_missed": sorted(leaks - flagged),
        "post_event_false_alarms": sorted(flagged - leaks - {case["target"]}),
        "identifier_or_constant_flagged": sorted(traps_id & (suggested_exclude | flagged)),
        "fields_proposed_by_llm": proposed,
    }


def _simulate_person(draft, case: dict[str, Any], profile: DatasetProfileV1) -> int:
    """Decisiones correctas documentadas; devuelve cuántas confirmaciones hizo la persona."""

    before = draft.revision
    answers = {
        DraftField.DECISION_ACTOR_LABEL: ACTOR,
        DraftField.OBJECTIVE: "Priorizar a quién atender primero con los datos de prueba",
        DraftField.TARGET_COLUMN: case["target"],
        DraftField.POSITIVE_CLASS: "sí",
        DraftField.PREDICTION_MOMENT: case["moment"],
        DraftField.FALSE_POSITIVE_CONSEQUENCE: "Se atiende un caso que no lo necesitaba",
        DraftField.FALSE_NEGATIVE_CONSEQUENCE: "El evento ocurre sin haberlo anticipado",
        DraftField.PRIMARY_METRIC: case["metric"],
        DraftField.METRIC_RATIONALE: "Refleja el costo relativo de los errores declarados",
        DraftField.GUARDRAIL_MODE: "no_quantitative_guardrail",
    }
    for name, value in answers.items():
        current = draft.get(name)
        if current.value == value and current.source is FieldSource.LLM:
            draft.confirm(name, actor_label=ACTOR)
        else:
            draft.set_by_person(name, value, actor_label=ACTOR)
    unsafe = {c.name for c in profile.columns if c.possible_identifier or c.constant}
    for item in draft.candidate_features():
        leak = item.column in case["post_event"]
        draft.set_feature(
            item.column,
            use=DraftUse.EXCLUDE if leak or item.column in unsafe else DraftUse.INCLUDE,
            available_at_prediction=Availability.NO if leak else Availability.YES,
            actor_label=ACTOR,
        )
    return draft.revision - before


def measure(name: str, case: dict[str, Any], provider, tmp: Path) -> dict[str, Any]:
    connection = get_connection(tmp / f"{name}.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    storage = tmp / f"store-{name}"
    out: dict[str, Any] = {"dataset": name}

    started = time.perf_counter()
    project_id, dataset_id = _ingest(repository, storage, name)
    out["seconds_ingest_and_profile"] = round(time.perf_counter() - started, 2)
    dataset = repository.get_dataset(dataset_id)
    profile = DatasetProfileV1.model_validate(json.loads(dataset["profile_json"]))
    out["rows"], out["columns"] = profile.row_count, profile.column_count

    service = ContextDraftService(repository, storage)
    stored = service.open(project_id, dataset_id)
    review = review_outgoing(case["message"], SafeProfileForLLMV1.from_profile(profile))
    started = time.perf_counter()
    turn = ContextAssistant(provider).next_turn(review, stored.draft, profile)
    out["seconds_assistant"] = round(time.perf_counter() - started, 2)
    apply_turn(stored.draft, turn.turn, profile)
    out["assistant"] = {
        "mode": turn.mode,
        "fallback_reason": turn.reason,
        "input_tokens": turn.trace.get("input_tokens"),
        "output_tokens": turn.trace.get("output_tokens"),
        "questions": len(turn.turn.questions),
        "column_refs_sent": len(column_refs(profile)),
        **_assistant_quality(stored.draft, case, profile),
    }

    confirmations = _simulate_person(stored.draft, case, profile)
    context = service.finalize(stored)
    out["human"] = {
        "confirmations_in_context": confirmations,
        "variables_reviewed": len(stored.draft.candidate_features()),
        "formal_approvals": 2,
    }

    strategy = StrategyService(repository).create_template_version(
        project_id=project_id, context_version_id=context.context_version_id
    )
    explanation = StrategyExplainer(provider).explain(
        StrategyProposalV1.model_validate(strategy.proposal.model_dump())
    )
    repository.decide_training(
        strategy_version_id=strategy.strategy_version_id,
        decision="APPROVED",
        actor_label=ACTOR,
    )
    started = time.perf_counter()
    result = MLRunService(repository, storage).execute(strategy.strategy_version_id)
    out["seconds_training_and_evaluation"] = round(time.perf_counter() - started, 2)
    evaluation = result.evaluation
    holdout = evaluation.holdout
    prevalence = holdout.positive_support / holdout.support if holdout else None
    out["ml"] = {
        "models_trained": len(evaluation.validation),
        "primary_metric": evaluation.selection.primary_metric,
        "baseline_validation": round(evaluation.selection.baseline_value, 3),
        "selected_model": result.selected_model,
        "selected_validation": (
            round(evaluation.validation[result.selected_model].primary_metric_value, 3)
            if result.selected_model
            else None
        ),
        "holdout_rows": holdout.support if holdout else None,
        "holdout_prevalence": round(prevalence, 3) if prevalence else None,
        "holdout_average_precision": round(holdout.average_precision, 3) if holdout else None,
        "holdout_precision": round(holdout.precision, 3) if holdout else None,
        "holdout_recall": round(holdout.recall, 3) if holdout else None,
        "precision_lift_vs_random": (
            round(holdout.precision / prevalence, 2) if holdout and prevalence else None
        ),
        "confusion_matrix": holdout.confusion_matrix if holdout else None,
        "top_features": [item.feature for item in evaluation.permutation_importance[:3]],
    }

    narrative = NarrativeService(provider) if provider else None
    started = time.perf_counter()
    _, report_mode, report_reason = ReportService(repository, narrative=narrative).create_report(
        result.run_id
    )
    out["seconds_report"] = round(time.perf_counter() - started, 2)
    out["llm_other"] = {
        "strategy_explanation": explanation.source,
        "strategy_explanation_reason": explanation.trace.get("fallback_reason"),
        "report_narrative": report_mode,
        "report_narrative_reason": report_reason,
    }
    connection.close()
    return out


def _summary(results: list[dict[str, Any]]) -> None:
    print("\n| Dataset | IA: resultado | IA: post-evento | ML vs azar | Recall | Confirmaciones |")
    print("|---|---|---|---|---|---|")
    for item in results:
        assistant, ml, human = item["assistant"], item["ml"], item["human"]
        found = len(assistant["post_event_found"])
        total = found + len(assistant["post_event_missed"])
        print(
            f"| {item['dataset']} | {'sí' if assistant['target_correct'] else 'no'} "
            f"({assistant['mode']}) | {found}/{total} "
            f"(+{len(assistant['post_event_false_alarms'])} falsas) | "
            f"x{ml['precision_lift_vs_random']} | {ml['holdout_recall']} | "
            f"{human['confirmations_in_context']} + 2 aprobaciones |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="Usar el proveedor de .env")
    parser.add_argument("--json", type=Path, help="Guardar resultados en este archivo")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    settings = LLMSettings.from_env(env_file=ROOT / ".env")
    provider = build_provider(settings) if args.live else None
    print(f"IA generativa: {'activa · ' + settings.model if provider else 'desactivada'}")

    results = []
    with tempfile.TemporaryDirectory(
        prefix="goal_settings-measure-", ignore_cleanup_errors=True
    ) as tmp:
        for name, case in CASES.items():
            print(f"Midiendo {name}…")
            results.append(measure(name, case, provider, Path(tmp)))
    _summary(results)
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResultados completos en {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

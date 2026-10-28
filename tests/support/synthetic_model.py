"""Modelo aprobado y publicado sobre `mantenimiento_maquinas.csv` para probar su uso."""

from pathlib import Path

from core.data_engine import DataIngestionService
from core.db.connection import get_connection
from core.db.migrations import migrate
from core.db.repositories import WorkflowRepository
from core.domain.context_service import BusinessContextService
from core.domain.strategy import StrategyProposalV1
from core.domain.strategy_service import StrategyService
from core.ml_engine import MLRunService
from core.ml_engine.scoring import ScoringService
from core.reporting import ReportService

ROOT = Path(__file__).parents[2]
SYNTHETIC = ROOT / "data" / "synthetic"
VARIANTS = SYNTHETIC / "variantes"
FEATURES = (
    "tipo_maquina",
    "temperatura_aire_c",
    "temperatura_proceso_c",
    "velocidad_rpm",
    "torque_nm",
    "desgaste_herramienta_min",
    "turno_noche",
)
EXCLUDED = {
    "id_maquina": (True, "Identificador"),
    "planta": (True, "Constante"),
    "horas_detenida_post_falla": (False, "Se conoce después de la falla"),
}


def approved_maintenance_model(tmp_path: Path) -> tuple[WorkflowRepository, ScoringService, str]:
    connection = get_connection(tmp_path / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    storage_root = tmp_path / "runtime"
    project_id = repository.create_project("Mantenimiento sintético")
    admission = DataIngestionService(repository, storage_root).ingest_bytes(
        project_id=project_id,
        request={
            "original_filename_display": "mantenimiento_maquinas.csv",
            "source_name": "Generado por scripts/generate_synthetic_datasets.py (datos falsos)",
            "license": "TEST-ONLY",
            "classification": "synthetic",
            "authorization_status": "approved",
            "anonymization_declared": False,
        },
        content=(SYNTHETIC / "mantenimiento_maquinas.csv").read_bytes(),
    )
    decisions = [
        {
            "column": column,
            "use": "include",
            "available_at_prediction": True,
            "reason": "Disponible al inicio del turno",
        }
        for column in FEATURES
    ] + [
        {
            "column": column,
            "use": "exclude",
            "available_at_prediction": available,
            "reason": reason,
        }
        for column, (available, reason) in EXCLUDED.items()
    ]
    context = BusinessContextService(repository, storage_root).create_version(
        project_id=project_id,
        dataset_id=admission.metadata.dataset_id,
        context={
            "objective": "Priorizar máquinas a revisar al inicio del turno",
            "target_column": "falla_maquina",
            "positive_class": "sí",
            "prediction_moment": "Al inicio del turno, antes de una falla",
            "false_positive_consequence": "Revisión innecesaria",
            "false_negative_consequence": "Falla no anticipada",
            "primary_metric": "f1",
            "metric_rationale": "Equilibrio entre revisar de más y no anticipar fallas",
            "decision_actor_label": "Responsable de prueba",
            "feature_decisions": decisions,
            "guardrails": {"mode": "no_quantitative_guardrail", "rules": []},
        },
    )
    strategy_service = StrategyService(repository)
    template = strategy_service.build_template(
        project_id=project_id,
        context_version_id=context.context_version_id,
    ).model_dump(mode="json")
    template["model_params"]["random_forest"]["n_estimators"] = 30
    strategy = strategy_service.create_version(
        project_id=project_id,
        proposal=StrategyProposalV1.model_validate(template),
    )
    repository.decide_training(
        strategy_version_id=strategy.strategy_version_id,
        decision="APPROVED",
        actor_label="Responsable de prueba",
    )
    result = MLRunService(repository, storage_root).execute(strategy.strategy_version_id)
    report_id = ReportService(repository).create_template_report(result.run_id)
    repository.decide_publication(
        report_id=report_id,
        decision="APPROVED",
        actor_label="Responsable de prueba",
    )
    repository.publish_report(report_id)
    return repository, ScoringService(repository, storage_root), result.run_id

"""GA-06: un análisis se guarda como tabla con receta y puede elegirse para entrenar."""

import json
import sqlite3
from pathlib import Path

import pytest

from core.analysis.contracts import AnalysisPlanV1
from core.analysis.engine import load_project_source
from core.analysis.tables import AnalysisTableService
from core.analysis.validation import AnalysisPlanError
from core.data_engine import analysis_schema, validate_and_profile
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from tests.support.domains import DOMAINS
from tests.unit.test_catalog import ingest


@pytest.fixture
def repository(tmp_path: Path):
    connection = get_connection(tmp_path / "goal_settings.sqlite3")
    migrate(connection)
    yield WorkflowRepository(connection)
    connection.close()


def _row_level_plan(schema: dict[str, str]) -> AnalysisPlanV1:
    """Una fila por caso, sin personales, con un resultado sí/no creado desde un número."""

    kept = [name for name, kind in schema.items() if kind != "personal"]
    number = next(name for name in kept if schema[name] == "number")
    return AnalysisPlanV1.model_validate(
        {
            "sources": ["archivo"],
            "steps": [
                {"op": "select", "columns": kept},
                {
                    "op": "threshold_label",
                    "column": number,
                    "output_column": "resultado_marcado",
                    "percentile": 0.5,
                },
            ],
            "output": {"chart": "table"},
        }
    )


def _first_with_number(domain) -> Path:
    return next(
        path
        for path in domain.files
        if "number"
        in analysis_schema(validate_and_profile(path.name, path.read_bytes()).model_dump()).values()
    )


@pytest.mark.parametrize("domain", DOMAINS, ids=lambda domain: domain.key)
def test_every_business_saves_a_table_with_its_recipe(
    domain, repository: WorkflowRepository, tmp_path: Path
) -> None:
    project_id = repository.create_project("Tabla")
    path = _first_with_number(domain)
    source_id = ingest(repository, tmp_path, project_id, path)
    schema = analysis_schema(json.loads(repository.get_dataset(source_id)["profile_json"]))
    plan = _row_level_plan(schema)

    saved = AnalysisTableService(repository, tmp_path).save(
        project_id=project_id,
        plan=plan,
        source_ids={"archivo": source_id},
        display_name="  Tabla   preparada ",
        actor_label="Analista",
    )

    dataset = repository.get_dataset(saved.dataset_id)
    assert dataset["dataset_kind"] == "DERIVED" and not dataset["is_active"]
    assert dataset["original_filename_display"] == "Tabla preparada.csv"
    assert saved.rows == repository.get_dataset(source_id)["n_rows"]  # no se pierden filas
    recipe = repository.get_analysis_table(saved.dataset_id)
    assert json.loads(recipe["sources_json"])[0]["sha256"] == dataset_sha(repository, source_id)
    assert recipe["plan_sha256"] == saved.plan_sha256

    # La tabla se puede volver a analizar y elegir para entrenar (decisión humana aparte).
    reloaded = load_project_source(repository, tmp_path, saved.dataset_id)
    assert set(reloaded.frame["resultado_marcado"].dropna().unique()) <= {"no", "sí"}
    repository.set_modeling_table(project_id, saved.dataset_id, actor_label="Analista")
    assert repository.get_dataset(saved.dataset_id)["is_active"]


def dataset_sha(repository: WorkflowRepository, dataset_id: str) -> str:
    return str(repository.get_dataset(dataset_id)["sha256"])


def _project_with_source(repository: WorkflowRepository, tmp_path: Path):
    domain = next(item for item in DOMAINS if item.has_outcome)
    project_id = repository.create_project("Tabla")
    path = _first_with_number(domain)
    source_id = ingest(repository, tmp_path, project_id, path)
    schema = analysis_schema(json.loads(repository.get_dataset(source_id)["profile_json"]))
    return project_id, source_id, schema


def test_name_actor_and_empty_results_are_explained(
    repository: WorkflowRepository, tmp_path: Path
) -> None:
    project_id, source_id, schema = _project_with_source(repository, tmp_path)
    service = AnalysisTableService(repository, tmp_path)
    plan = _row_level_plan(schema)
    common = {"project_id": project_id, "plan": plan, "source_ids": {"archivo": source_id}}

    with pytest.raises(AnalysisPlanError) as unnamed:
        service.save(**common, display_name=" ", actor_label="Analista")
    assert unnamed.value.code == "TABLE_NAME_REQUIRED"
    with pytest.raises(AnalysisPlanError) as anonymous:
        service.save(**common, display_name="Tabla", actor_label=" ")
    assert anonymous.value.code == "TABLE_ACTOR_REQUIRED"

    number = next(name for name, kind in schema.items() if kind == "number")
    empty = AnalysisPlanV1.model_validate(
        {
            "sources": ["archivo"],
            "steps": [{"op": "filter", "column": number, "operator": "is_null"}],
            "output": {"chart": "table"},
        }
    )
    with pytest.raises(AnalysisPlanError) as nothing:
        service.save(**{**common, "plan": empty}, display_name="Vacía", actor_label="Analista")
    assert nothing.value.code in {"TABLE_EMPTY", "TABLE_NOT_VALID"}
    assert not list((tmp_path / "data" / "accepted").glob("*.tmp"))


def test_recipe_is_append_only_and_audited(repository: WorkflowRepository, tmp_path: Path) -> None:
    project_id, source_id, schema = _project_with_source(repository, tmp_path)
    saved = AnalysisTableService(repository, tmp_path).save(
        project_id=project_id,
        plan=_row_level_plan(schema),
        source_ids={"archivo": source_id},
        display_name="Tabla",
        actor_label="Analista",
    )

    with pytest.raises(sqlite3.DatabaseError):
        repository.connection.execute("DELETE FROM analysis_tables")
    events = repository.connection.execute(
        "SELECT detail_json FROM audit_events WHERE event_type = 'ANALYSIS_TABLE_SAVED'"
    ).fetchall()
    assert json.loads(events[0]["detail_json"])["plan_sha256"] == saved.plan_sha256

"""GA-04c: «Pregúntale a tus datos» con IA falsa, sin IA y con preguntas imposibles."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.ai_engine import LLMResponse, register_provider
from core.ai_engine.data_questions import _STEP_FIELDS
from core.data_engine import analysis_schema, validate_and_profile
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
from tests.support.domains import DOMAINS
from tests.unit.test_catalog import ingest

APP_PATH = Path(__file__).parents[2] / "app" / "main.py"
DOMAIN = next(item for item in DOMAINS if item.multi_file and item.has_outcome)
FILE = min(
    DOMAIN.files, key=lambda path: path.stat().st_size if path.stat().st_size > 50_000 else 10**9
)
SCHEMA = analysis_schema(validate_and_profile(FILE.name, FILE.read_bytes()).model_dump())
CATEGORY = next(name for name, kind in SCHEMA.items() if kind == "category")
IDENTIFIER = next(name for name, kind in SCHEMA.items() if kind == "identifier")
SOURCE = FILE.stem.lower()


class FakeProvider:
    name, model = "pytest_questions", "fake"
    requests: list = []
    plan_answer: dict = {}

    def complete_json(self, request):
        FakeProvider.requests.append(request)
        if request.schema_name == "analysis_plan":
            content = FakeProvider.plan_answer
        else:
            content = {
                "explanation": "Hay grupos con más casos que otros.",
                "findings": ["El resultado compara la cantidad de casos por grupo."],
                "recommendation": "Revisar primero el grupo con más casos.",
                "caveats": ["Una diferencia no prueba una causa."],
            }
        return LLMResponse(
            content=json.dumps(content), provider=self.name, model=self.model, latency_ms=1
        )


def _step(op: str, **fields) -> dict:
    step = dict.fromkeys(_STEP_FIELDS)
    step.update(op=op, **fields)
    return step


COUNT_PLAN = {
    "answerable": True,
    "reason": "",
    "sources": [SOURCE],
    "steps": [
        _step(
            "group_by",
            by=[CATEGORY],
            aggregations=[{"function": "count", "column": None, "output_column": "casos"}],
        ),
        _step("sort", by=["casos"], descending=True),
    ],
    "output": {"chart": "bar", "x": CATEGORY, "y": "casos", "color": None},
}


@pytest.fixture
def runtime_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("GOAL_SETTINGS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GOAL_SETTINGS_ENV_FILE", str(tmp_path / "missing.env"))
    for name in ("LLM_ENABLED", "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    connection = get_connection(tmp_path / "goal_settings.sqlite3")
    migrate(connection)
    repository = WorkflowRepository(connection)
    ingest(repository, tmp_path, repository.create_project("Preguntas"), FILE)
    connection.close()
    yield tmp_path


def _with_fake_ai(monkeypatch: pytest.MonkeyPatch, plan: dict) -> None:
    register_provider("pytest_questions", lambda settings: FakeProvider())
    FakeProvider.requests = []
    FakeProvider.plan_answer = plan
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "pytest_questions")
    monkeypatch.setenv("LLM_MODEL", "fake")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9/v1")


def _ask(app: AppTest, question: str) -> AppTest:
    next(item for item in app.text_area if item.label == "Tu pregunta").input(question)
    next(item for item in app.button if item.label == "Preguntar a los datos").click()
    app.run(timeout=60)
    assert not app.exception, app.exception
    return app


def _audit(runtime_dir: Path) -> list[dict]:
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    rows = connection.execute(
        "SELECT detail_json FROM audit_events WHERE event_type = 'AI_DATA_SHARED'"
    ).fetchall()
    connection.close()
    return [json.loads(row["detail_json"]) for row in rows]


def test_question_is_planned_calculated_and_explained(
    runtime_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_fake_ai(monkeypatch, COUNT_PLAN)
    app = _ask(AppTest.from_file(str(APP_PATH)).run(timeout=60), "¿Qué grupo tiene más casos?")

    texts = " ".join(item.value for item in app.markdown)
    assert "Calculé cantidad de casos por" in texts
    assert app.get("arrow_vega_lite_chart") or app.get("vega_lite_chart")
    assert any("Hay grupos con más casos" in item.value for item in app.get("markdown"))
    assert "Revisar primero el grupo con más casos." in texts
    sent = " ".join(request.user for request in FakeProvider.requests)
    assert IDENTIFIER not in sent  # oculto por defecto: nunca sale
    purposes = {entry["purpose"] for entry in _audit(runtime_dir)}
    assert purposes == {"plan_question", "interpret_result"}


def test_impossible_questions_explain_why(
    runtime_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_fake_ai(
        monkeypatch,
        {
            "answerable": False,
            "reason": "Estos datos no traen esa información.",
            "sources": [],
            "steps": [],
            "output": {"chart": "table", "x": None, "y": None, "color": None},
        },
    )
    app = _ask(AppTest.from_file(str(APP_PATH)).run(timeout=60), "¿Lloverá mañana?")

    assert any("Estos datos no traen esa información." in item.value for item in app.info)


def test_without_ai_a_suggested_question_still_answers(runtime_dir: Path) -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    suggestion = next(item for item in app.button if (item.key or "").endswith("-suggestion-0"))
    suggestion.click()
    app.run(timeout=60)
    next(item for item in app.button if item.label == "Preguntar a los datos").click()
    app.run(timeout=60)

    assert not app.exception, app.exception
    assert app.get("arrow_vega_lite_chart") or app.get("vega_lite_chart")
    assert any("mayor valor" in item.value for item in app.get("markdown"))
    assert _audit(runtime_dir) == []  # sin IA no sale nada


def test_answer_is_saved_as_a_table_to_train(
    runtime_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_fake_ai(monkeypatch, COUNT_PLAN)
    app = _ask(AppTest.from_file(str(APP_PATH)).run(timeout=60), "¿Qué grupo tiene más casos?")
    next(item for item in app.text_input if item.label == "Nombre de la tabla").input("Grupos")
    next(item for item in app.text_input if item.label == "¿Quién la guarda?").input("Analista")
    next(item for item in app.checkbox if item.label.startswith("Usarla para entrenar")).check()
    next(item for item in app.button if item.label == "Guardar tabla").click()
    app.run(timeout=60)

    assert not app.exception, app.exception
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    table = connection.execute(
        "SELECT d.is_active, d.original_filename_display FROM analysis_tables t "
        "JOIN datasets d ON d.id = t.output_dataset_id"
    ).fetchone()
    connection.close()
    # V2-11: un resumen por grupos se guarda, pero no se activa para entrenar sin revisión.
    assert table["original_filename_display"] == "Grupos.csv" and table["is_active"] == 0
    assert any("no la activé para entrenar" in item.value for item in app.info)


def test_relations_between_files_are_listed(runtime_dir: Path) -> None:
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    repository = WorkflowRepository(connection)
    project_id = repository.list_active_projects()[0]["id"]
    for path in DOMAIN.files:
        if path != FILE:
            ingest(repository, runtime_dir, project_id, path)
    connection.close()
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    next(item for item in app.button if item.label == "Buscar relaciones").click()
    app.run(timeout=60)

    assert not app.exception, app.exception
    assert any("encaja con" in item.value for item in app.markdown)


def test_automatic_analyses_render(runtime_dir: Path) -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    next(item for item in app.button if item.label == "Ver análisis").click()
    app.run(timeout=60)

    assert not app.exception, app.exception
    assert app.get("arrow_vega_lite_chart") or app.get("vega_lite_chart")


def test_proposals_without_ai_come_from_column_types(runtime_dir: Path) -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    next(item for item in app.button if item.label == "Pedir propuestas").click()
    app.run(timeout=60)

    assert not app.exception, app.exception
    assert any("sin IA" in item.value for item in app.caption)
    assert any("Preguntas para hacerle" in item.value for item in app.markdown)
    assert _audit(runtime_dir) == []  # sin IA no sale nada


def test_business_notes_are_versioned_and_reach_the_ai(
    runtime_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_fake_ai(monkeypatch, COUNT_PLAN)
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    next(item for item in app.text_area if item.label == "Notas").input("Cada fila es un caso.")
    next(item for item in app.text_input if item.label == "¿Quién las escribe?").input("Analista")
    next(item for item in app.button if item.label == "Guardar notas").click()
    app.run(timeout=60)
    app = _ask(app, "¿Qué grupo tiene más casos?")

    plan_request = next(r for r in FakeProvider.requests if r.schema_name == "analysis_plan")
    assert "Cada fila es un caso." in plan_request.user
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    assert connection.execute("SELECT COUNT(*) FROM project_notes").fetchone()[0] == 1
    connection.close()


def test_quality_review_renders(runtime_dir: Path) -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    next(item for item in app.radio if item.label == "Qué quieres ver").set_value("quality")
    app.run(timeout=60)
    next(item for item in app.button if item.label == "Ver análisis").click()
    app.run(timeout=60)

    assert not app.exception, app.exception
    assert any("Revisión de calidad" in item.value for item in app.markdown)


def test_a_decision_is_recorded_from_the_screen(runtime_dir: Path) -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    next(item for item in app.text_area if item.label == "Qué decidiste").input(
        "Revisar primero el grupo con más casos"
    )
    next(item for item in app.text_input if item.label == "¿Quién decide?").input("Analista")
    next(item for item in app.button if item.label == "Guardar decisión").click()
    app.run(timeout=60)

    assert not app.exception, app.exception
    assert any("Revisar primero el grupo" in item.value for item in app.markdown)


def test_incoming_events_become_a_project_file(runtime_dir: Path) -> None:
    from datetime import UTC, datetime

    from core.integration.events import EventIngestionService

    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    repository = WorkflowRepository(connection)
    project_id = repository.list_active_projects()[0]["id"]
    events = [
        {
            "source": "sistema_a",
            "entity_type": "caso",
            "entity_id": f"c{index}",
            "occurred_at": datetime(2026, 9, 1, index % 24, tzinfo=UTC),
            "attributes": {"medida": index * 1.5, "grupo": "ab"[index % 2]},
        }
        for index in range(30)
    ]
    EventIngestionService(repository, runtime_dir).ingest(
        project_id, events, source_label="Sistema A", actor_label="Integración"
    )
    connection.close()
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    next(item for item in app.text_input if item.label == "¿Quién lo convierte?").input("Analista")
    next(item for item in app.button if item.label == "Convertir en archivo").click()
    app.run(timeout=60)

    assert not app.exception, app.exception
    connection = get_connection(runtime_dir / "goal_settings.sqlite3")
    sources = connection.execute(
        "SELECT COUNT(*) FROM datasets WHERE source = 'observation_events:caso'"
    ).fetchone()[0]
    connection.close()
    assert sources == 1

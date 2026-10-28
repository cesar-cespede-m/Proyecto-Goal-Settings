"""C2-07: resumen del ciclo con próximos pasos, sólo de lectura."""

from core.domain.cycle_summary import cycle_summary
from core.domain.decisions import DecisionService
from tests.unit.test_recipe_retraining import trained  # noqa: F401


def test_summary_counts_models_decisions_and_suggests_steps(trained) -> None:  # noqa: F811
    repository, storage, run_id = trained
    project_id = repository.get_run_project_id(run_id)
    DecisionService(repository).record(
        project_id,
        decision="Revisar primero los casos marcados",
        based_on="report",
        reference_id=run_id,
        actor_label="Analista",
    )

    summary = cycle_summary(repository, storage, project_id)

    assert [model.run_id for model in summary.models_in_use] == [run_id]
    assert (summary.decisions, summary.decisions_with_outcome) == (1, 0)
    assert any("sin resultado anotado" in step for step in summary.next_steps)
    assert summary.accepted_files >= 3  # dos fuentes y la tabla preparada

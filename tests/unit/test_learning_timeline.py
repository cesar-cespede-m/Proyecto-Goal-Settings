"""V2-28: línea de tiempo del aprendizaje a través de los reentrenamientos."""

from core.domain.learning_timeline import learning_timeline
from core.domain.retraining_service import RetrainingService
from tests.unit.test_recipe_retraining import _files, trained  # noqa: F401


def test_timeline_follows_the_retraining_chain(trained) -> None:  # noqa: F811
    repository, storage, run_id = trained
    origin = repository.get_run_project_id(run_id)
    started = RetrainingService(repository, storage).start_from_recipe(
        source_run_id=run_id, files=_files(2), actor_label="Analista", data_authorized=True
    )

    from_origin = learning_timeline(repository, storage, origin)
    from_new = learning_timeline(repository, storage, started.project_id)

    assert [entry.run_id for entry in from_origin.entries] == [run_id]
    assert from_origin.entries[0].approved and from_origin.entries[0].metric
    assert from_new.entries == from_origin.entries  # la nueva aún no entrena: misma historia
    assert "etapas de aprendizaje" in from_new.summary

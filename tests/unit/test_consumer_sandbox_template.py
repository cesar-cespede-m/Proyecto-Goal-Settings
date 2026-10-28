"""SDLC-02.1: estructura segura del repositorio consumidor antes de publicarlo."""

from pathlib import Path

from core.integration import load_public_key, public_key_id

ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / "templates" / "goal_settings-consumer-sandbox"


def test_consumer_sandbox_template_has_public_material_only() -> None:
    public_key_path = TEMPLATE / "goal_settings" / "public-key.pem"

    assert (TEMPLATE / "README.md").is_file()
    assert (TEMPLATE / "model-usage" / "README.md").is_file()
    assert public_key_path.is_file()
    assert public_key_id(load_public_key(public_key_path)) == (
        "1c2e6069d959610c8065b5d73ff6a754e44248214fe14f2a6343e445a5b2a4a2"
    )
    assert not (TEMPLATE / "release-evidence.json").exists()
    assert not list(TEMPLATE.rglob("*private*"))
    assert "release-evidence.json" in (TEMPLATE / ".gitignore").read_text(encoding="utf-8")

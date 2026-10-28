"""Contrato estático del contenedor local; no reemplaza un build Docker real."""

from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_docker_runs_non_root_with_healthcheck_and_runtime_env() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in dockerfile
    assert "USER appuser" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "/_stcore/health" in dockerfile
    assert "GOAL_SETTINGS_RUNTIME_DIR=/app/runtime" in dockerfile
    assert "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false" in dockerfile
    assert "COPY .streamlit ./.streamlit" in dockerfile
    assert 'CMD ["streamlit", "run", "app/main.py"' in dockerfile


def test_compose_binds_only_loopback_persists_runtime_and_loads_optional_env() -> None:
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert '"127.0.0.1:8501:8501"' in compose
    assert "./runtime:/app/runtime" in compose
    assert "env_file:" in compose
    assert "path: .env" in compose
    assert "required: false" in compose
    assert 'LLM_ENABLED: "false"' not in compose
    assert "no-new-privileges:true" in compose


def test_docker_context_excludes_secrets_runtime_and_raw_samples() -> None:
    ignored = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())
    assert {".env", ".env.*", "runtime", "backups", "data/samples", "*.pem", "*.key"} <= ignored


def test_env_example_only_documents_variables_the_code_reads() -> None:
    """UX-02: una variable documentada que nadie lee hace creer que se aisló la base."""
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    keys = [
        line.split("=", 1)[0].strip()
        for line in example.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    ]
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for folder in ("app", "core")
        for path in (ROOT / folder).rglob("*.py")
    )
    # OPENAI_API_KEY se lee de forma indirecta mediante LLM_API_KEY_ENV.
    unread = [key for key in keys if key != "OPENAI_API_KEY" and f'"{key}"' not in sources]
    assert "GOAL_SETTINGS_RUNTIME_DIR" in keys
    assert unread == []


def test_ci_runs_real_docker_smoke_with_restart_and_persistence() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    smoke = (ROOT / "scripts/docker_smoke.sh").read_text(encoding="utf-8")
    assert "docker-smoke:" in workflow
    assert "docker compose config --quiet" in workflow
    assert "bash scripts/docker_smoke.sh" in workflow
    assert 'docker build --tag "$image_name" .' in smoke
    assert 'docker restart "$container_name"' in smoke
    assert 'docker exec "$container_name" test -s /app/runtime/goal_settings.sqlite3' in smoke
    assert "SELECT COUNT(*) FROM projects" in smoke
    assert "no-new-privileges:true" in smoke

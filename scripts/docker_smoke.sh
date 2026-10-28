#!/usr/bin/env bash
set -euo pipefail

image_name="${1:-goal_settings-mvp:ci}"
container_name="goal_settings-smoke-${GITHUB_RUN_ID:-$$}"
runtime_dir="$(mktemp -d)"
host_port="18501"

cleanup() {
  exit_code=$?
  if [ "$exit_code" -ne 0 ]; then
    docker logs "$container_name" 2>/dev/null || true
  fi
  docker rm --force "$container_name" >/dev/null 2>&1 || true
  exit "$exit_code"
}
trap cleanup EXIT

wait_for_http() {
  for _ in $(seq 1 60); do
    if curl --fail --silent --show-error \
      "http://127.0.0.1:${host_port}/_stcore/health" >/dev/null; then
      return 0
    fi
    sleep 2
  done
  echo "Streamlit no respondió dentro del plazo" >&2
  return 1
}

wait_for_container_health() {
  for _ in $(seq 1 45); do
    status="$(docker inspect --format '{{.State.Health.Status}}' "$container_name")"
    if [ "$status" = "healthy" ]; then
      return 0
    fi
    if [ "$status" = "unhealthy" ]; then
      echo "El HEALTHCHECK de la imagen informó unhealthy" >&2
      return 1
    fi
    sleep 2
  done
  echo "El HEALTHCHECK de la imagen no llegó a healthy" >&2
  return 1
}

docker build --tag "$image_name" .
sudo chown 10001:10001 "$runtime_dir"
docker run --detach \
  --name "$container_name" \
  --publish "127.0.0.1:${host_port}:8501" \
  --mount "type=bind,source=${runtime_dir},target=/app/runtime" \
  --security-opt no-new-privileges:true \
  --env GOAL_SETTINGS_RUNTIME_DIR=/app/runtime \
  --env LLM_ENABLED=false \
  "$image_name" >/dev/null

echo "Esperando HEALTHCHECK inicial..."
wait_for_container_health
wait_for_http

echo "Creando estado de prueba persistente..."
docker exec "$container_name" python -c '
from core.db import get_connection, migrate
from core.db.repositories import WorkflowRepository
connection = get_connection("/app/runtime/goal_settings.sqlite3")
migrate(connection)
print(WorkflowRepository(connection).create_project("Fixture smoke Docker CI"))
connection.close()
'
docker exec "$container_name" test -s /app/runtime/goal_settings.sqlite3

echo "Reiniciando el contenedor..."
docker restart "$container_name" >/dev/null
wait_for_container_health
wait_for_http

echo "Verificando la persistencia SQLite tras el reinicio..."
docker exec "$container_name" python -c '
from core.db import get_connection
connection = get_connection("/app/runtime/goal_settings.sqlite3")
count = connection.execute(
    "SELECT COUNT(*) FROM projects WHERE name = ?",
    ("Fixture smoke Docker CI",),
).fetchone()[0]
connection.close()
assert count == 1, f"Se esperó un proyecto persistido y se observaron {count}"
'

echo "Docker smoke PASS: build, healthcheck, restart y persistencia SQLite."

"""Instala y abre Goal Settings desde una copia local, sin privilegios de administrador."""

from __future__ import annotations

import argparse
import os
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def environment_python(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def environment_ready(python: Path) -> bool:
    if not python.is_file():
        return False
    code = """
import sys
from importlib.metadata import version
from pathlib import Path
import tomllib
assert (3, 12) <= sys.version_info[:2] < (3, 15)
for line in Path('requirements.lock').read_text(encoding='utf-8').splitlines():
    if line.strip() and not line.startswith('#'):
        name, expected = line.strip().split('==')
        assert version(name) == expected, name
assert version('proyecto-goal-settings') == tomllib.loads(
    Path('pyproject.toml').read_text(encoding='utf-8'))['project']['version']
import streamlit, sklearn, core
"""
    result = subprocess.run([str(python), "-c", code], cwd=ROOT, capture_output=True, text=True)
    return result.returncode == 0


def install(directory: Path) -> Path:
    python = environment_python(directory)
    if not python.exists():
        print("Creando un entorno Python privado...", flush=True)
        venv.EnvBuilder(with_pip=True).create(directory)
    print("Instalando versiones fijadas. La primera vez necesita Internet...", flush=True)
    run([str(python), "-m", "pip", "install", "-r", "requirements.lock"])
    run([str(python), "-m", "pip", "install", "-e", ".", "--no-deps"])
    run([str(python), "-m", "pip", "check"])
    if not environment_ready(python):
        raise RuntimeError("El entorno no coincide con requirements.lock. Revisa la instalación.")
    return python


def local_environment(runtime: Path, *, enable_ai: bool = False) -> dict[str, str]:
    env = dict(os.environ)
    env["GOAL_SETTINGS_RUNTIME_DIR"] = str(runtime.resolve())
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["PYTHONUTF8"] = "1"
    if not enable_ai:
        env["LLM_ENABLED"] = "false"
    return env


def prepare_demo(python: Path, directory: Path) -> Path:
    output = directory / "execution"
    runtime = output / "private-storage"
    target = runtime / "goal_settings.sqlite3"
    if target.exists():
        print("Abriendo tu copia de la demo; se conservan los cambios guardados.", flush=True)
        return runtime
    if output.exists():
        raise RuntimeError(
            "Hay una preparación incompleta. Usa --demo-dir con otra carpeta; "
            "no se borrará ni reemplazará la anterior."
        )
    print(
        "DEMO SINTÉTICA: entrenamientos reales, datos inventados y revisiones simuladas.\n"
        "Las cifras económicas son supuestos. No representa aprobaciones humanas ni ganancias.",
        flush=True,
    )
    env = local_environment(runtime)
    data = directory / "data"
    run([str(python), "-m", "scripts.generate_value_demo_datasets", "--output", str(data)], env=env)
    run(
        [
            str(python),
            "-m",
            "scripts.run_product_demo",
            "--data",
            str(data),
            "--output",
            str(output),
            "--simulate-reviews",
        ],
        env=env,
    )
    source = output / "goal_settings_demo.sqlite3"
    with sqlite3.connect(source) as src, sqlite3.connect(target) as dst:
        src.backup(dst)
    (runtime / "REVISIONES_SIMULADAS.txt").write_text(
        "Demo sintética con revisiones simuladas. No acredita aprobación humana ni impacto real.\n",
        encoding="utf-8",
    )
    return runtime


def ensure_free_port(port: int) -> None:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(
                f"El puerto {port} está ocupado. Cierra la otra instancia o usa --port {port + 1}."
            ) from error


def open_when_ready(url: str, process: subprocess.Popen) -> None:
    for _ in range(60):
        if process.poll() is not None:
            return
        try:
            with urllib.request.urlopen(url + "/_stcore/health", timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except OSError:
            pass
        time.sleep(0.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    parser.add_argument("--install-only", action="store_true")
    parser.add_argument("--check", action="store_true", help="Verifica sin instalar ni abrir")
    parser.add_argument("--demo", action="store_true", help="Demo con revisiones simuladas")
    parser.add_argument("--demo-dir", type=Path, default=ROOT / "runtime/reviewer-demo")
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / "runtime/local")
    parser.add_argument("--prepare-only", action="store_true", help="Prepara demo sin servidor")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--enable-ai", action="store_true", help="Respeta configuración IA local")
    args = parser.parse_args(argv)
    if not (3, 12) <= sys.version_info[:2] < (3, 15):
        parser.error("Usa Python 3.12, 3.13 o 3.14; esta entrega no admite Python 3.15.")
    if not 1024 <= args.port <= 65535:
        parser.error("El puerto debe estar entre 1024 y 65535.")
    if args.prepare_only and not args.demo:
        parser.error("--prepare-only requiere --demo.")
    if args.demo and args.enable_ai:
        parser.error("La demo preparada no utiliza proveedor IA. Abre el producto para usarlo.")
    try:
        python = environment_python(args.venv.resolve())
        ready = environment_ready(python)
        if args.check:
            if not ready:
                raise RuntimeError("Entorno incompleto. Ejecuta el lanzador con --install-only.")
            run([str(python), "-m", "pip", "check"])
            print("Goal Settings: Python, dependencias e importaciones correctos.")
            return 0
        if not ready:
            python = install(args.venv.resolve())
        if args.install_only:
            print("Instalación lista. Ejecuta INICIAR_GOAL_SETTINGS.cmd o INICIAR_DEMO.cmd.")
            return 0
        if not args.prepare_only:
            ensure_free_port(args.port)
        runtime = (
            prepare_demo(python, args.demo_dir.resolve())
            if args.demo
            else args.runtime_dir.resolve()
        )
        if args.prepare_only:
            print(f"Demo preparada en {runtime}")
            return 0
        url = f"http://127.0.0.1:{args.port}"
        print(
            f"\nGoal Settings: {url}\nDatos: {runtime}\nPara detener: Ctrl+C.\n",
            flush=True,
        )
        process = subprocess.Popen(
            [
                str(python),
                "-m",
                "streamlit",
                "run",
                "app/main.py",
                "--server.address=127.0.0.1",
                f"--server.port={args.port}",
                "--server.headless=true",
            ],
            cwd=ROOT,
            env=local_environment(runtime, enable_ai=args.enable_ai),
        )
        if not args.no_browser:
            threading.Thread(target=open_when_ready, args=(url, process), daemon=True).start()
        try:
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait(timeout=10)
            return 0
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(
            f"\nNo se pudo iniciar Goal Settings: {error}\nConsulta README.md.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

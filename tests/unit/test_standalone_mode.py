"""El producto autónomo no carga dependencias de la integración SDLC."""

import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_streamlit_app_imports_without_cryptography() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert not any(item.startswith("cryptography") for item in project["dependencies"])
    assert "cryptography==50.0.2" in project["optional-dependencies"]["sdlc"]

    program = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockCryptography(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "cryptography" or fullname.startswith("cryptography."):
                    raise ModuleNotFoundError("cryptography bloqueada para probar el modo autónomo")
                return None

        sys.meta_path.insert(0, BlockCryptography())
        import core.reporting
        import app.main
        print("standalone-import-ok")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "standalone-import-ok" in result.stdout

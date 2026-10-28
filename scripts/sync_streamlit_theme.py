"""Reescribe los colores de `.streamlit/config.toml` desde los tokens de app/ui/tokens.py."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ui.tokens import streamlit_theme  # noqa: E402

CONFIG = ROOT / ".streamlit" / "config.toml"


def synced(text: str) -> str:
    for key, value in streamlit_theme().items():
        line = f'{key} = "{value}"'
        pattern = re.compile(rf"^{key} = .*$", re.MULTILINE)
        if pattern.search(text):
            text = pattern.sub(line, text)
        else:
            text = text.replace("[theme]\n", f"[theme]\n{line}\n", 1)
    return text


def main() -> None:
    CONFIG.write_text(synced(CONFIG.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"Tema de Streamlit sincronizado en {CONFIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

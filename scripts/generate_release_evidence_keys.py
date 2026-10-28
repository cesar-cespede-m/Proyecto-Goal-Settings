"""Genera un par Ed25519 para firmar evidencia SDLC; la clave privada nunca va a Git."""

import argparse
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from core.integration import public_key_id  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument(
        "--allow-unencrypted-private-key",
        action="store_true",
        help="Sólo para un entorno local protegido; se recomienda una contraseña.",
    )
    args = parser.parse_args()

    for destination in (args.private_key, args.public_key):
        if destination.exists():
            raise SystemExit(f"No se sobrescribe una clave existente: {destination}")

    password = os.environ.get("GOAL_SETTINGS_SIGNING_KEY_PASSWORD")
    if password:
        encryption: serialization.KeySerializationEncryption = (
            serialization.BestAvailableEncryption(password.encode("utf-8"))
        )
    elif args.allow_unencrypted_private_key:
        encryption = serialization.NoEncryption()
    else:
        raise SystemExit(
            "Define GOAL_SETTINGS_SIGNING_KEY_PASSWORD o usa "
            "--allow-unencrypted-private-key sólo para una prueba local."
        )

    private_key = Ed25519PrivateKey.generate()
    args.private_key.parent.mkdir(parents=True, exist_ok=True)
    args.public_key.parent.mkdir(parents=True, exist_ok=True)
    args.private_key.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption,
        )
    )
    os.chmod(args.private_key, stat.S_IRUSR | stat.S_IWUSR)
    args.public_key.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    print(f"Clave pública creada. key_id={public_key_id(private_key.public_key())}")


if __name__ == "__main__":
    main()

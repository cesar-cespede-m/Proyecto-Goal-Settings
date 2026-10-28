"""Genera el caso B sintético de churn e-commerce; no usa datos de terceros.

Uso:
    python scripts/generate_case_b.py --output runtime/fixtures/case-b-churn.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_case_b(*, rows: int = 360, seed: int = 20261017) -> pd.DataFrame:
    """Devuelve un dataset binario reproducible con columnas y tipos distintos al caso A."""

    if rows < 100:
        raise ValueError(
            "El caso B requiere al menos 100 filas para sus particiones estratificadas"
        )
    rng = np.random.default_rng(seed)
    tenure_months = rng.integers(1, 49, rows)
    monthly_sessions = rng.poisson(7, rows)
    support_tickets = rng.poisson(1.4, rows)
    plan = rng.choice(["basic", "plus", "premium"], rows, p=[0.5, 0.35, 0.15])
    autopay = rng.choice([True, False], rows, p=[0.62, 0.38])
    score = (
        1.4 * (tenure_months <= 6)
        + 0.7 * (monthly_sessions <= 3)
        + 0.8 * (support_tickets >= 3)
        + 0.6 * (plan == "basic")
        + 0.8 * (~autopay)
        + rng.normal(0, 0.8, rows)
    )
    churn = (score >= 2.2).astype(int)
    return pd.DataFrame(
        {
            "customer_ref": [f"synthetic-customer-{index:04d}" for index in range(rows)],
            "tenure_months": tenure_months,
            "monthly_sessions": monthly_sessions,
            "support_tickets": support_tickets,
            "plan": plan,
            "autopay": autopay,
            "churn": churn,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=360)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = build_case_b(rows=args.rows, seed=args.seed)
    frame.to_csv(destination, index=False)
    print(
        f"Caso B sintético guardado en {destination} "
        f"({len(frame)} filas; positivos={int(frame['churn'].sum())})."
    )


if __name__ == "__main__":
    main()

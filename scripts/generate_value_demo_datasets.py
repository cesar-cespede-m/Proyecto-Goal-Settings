"""Genera dos recorridos económicos 100 % sintéticos con cronología reproducible."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

from core.analysis.snapshot import build_snapshot

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "synthetic" / "value_demo"
SEED = 20261023


def _recipe(kind: str) -> dict:
    features = [
        {"op": "count_events", "window_days": 2, "output_column": "activity_count"},
    ]
    if kind == "revenue":
        features.extend(
            [
                {
                    "op": "last_value",
                    "column": "initial_interest",
                    "output_column": "initial_interest",
                },
                {"op": "last_value", "column": "channel", "output_column": "channel"},
            ]
        )
        outcome = {
            "op": "not_reached_stage",
            "stage": {"column": "stage", "value": "purchase"},
            "within_days": 14,
            "output_column": "no_first_purchase",
        }
        availability = "label_available_at"
    else:
        features.extend(
            [
                {"op": "last_value", "column": "distance_km", "output_column": "distance_km"},
                {"op": "last_value", "column": "service_level", "output_column": "service_level"},
                {"op": "last_value", "column": "load_index", "output_column": "load_index"},
                {
                    "op": "last_value",
                    "column": "zone_quality",
                    "output_column": "zone_quality",
                },
            ]
        )
        outcome = {
            "op": "deadline_not_reached_stage",
            "stage": {"column": "stage", "value": "delivered"},
            "deadline_column": "promised_at",
            "output_column": "incumplimiento_promesa",
        }
        availability = "promised_at"
    return {
        "entity": "entity",
        "moment": "moment",
        "cutoff": {"mode": "per_entity_first_event_plus_days", "days": 2},
        "windows_days": [2],
        "features": features,
        "outcome": [outcome],
        "label_available_at_output_column": availability,
    }


def _events(
    kind: str, group: str, count: int, start: str, rng: random.Random
) -> tuple[pd.DataFrame, dict[str, bool]]:
    rows: list[dict] = []
    outcomes = {}
    base = pd.Timestamp(start, tz="UTC")
    for index in range(count):
        identifier = f"{kind[:3].upper()}-{group}-{index:04d}"
        first = base + pd.Timedelta(days=index if group in {"initial", "update"} else index // 8)
        engaged = index % 2 == 0
        noise = rng.random() < 0.10
        adverse = (not engaged) != noise
        outcomes[identifier] = adverse
        attributes = (
            {
                "initial_interest": 75 if engaged else 25,
                "channel": "app" if index % 3 else "web",
            }
            if kind == "revenue"
            else {
                "distance_km": 7 + index % 13,
                "service_level": "express" if index % 3 else "standard",
                "load_index": 25 if engaged else 75,
                "zone_quality": 90 if engaged else 30,
                "promised_at": (first + pd.Timedelta(days=9)).isoformat(),
            }
        )

        _add_event(
            rows, identifier, first, "registered" if kind == "revenue" else "ordered", attributes
        )
        for _ in range(3 if engaged else 1):
            _add_event(
                rows,
                identifier,
                first + pd.Timedelta(days=1),
                "activity" if kind == "revenue" else "check",
                attributes,
            )
        if not adverse:
            _add_event(
                rows,
                identifier,
                first + pd.Timedelta(days=5),
                "purchase" if kind == "revenue" else "delivered",
                attributes,
            )
        elif kind == "savings" and index % 5:
            _add_event(rows, identifier, first + pd.Timedelta(days=11), "delivered", attributes)
    return pd.DataFrame(rows), outcomes


def _coverage(start: str, end: str) -> dict:
    return {"observed_from": start, "observed_until": end, "complete": True}


def _economic(kind: str) -> dict:
    return {
        "currency": "CLP",
        "currency_minor_units": 0,
        "period": "mes",
        "planned_cases": 100,
        "capacity_cases": 20,
        "external_risk_rate": "0.40",
        "risk_rate_basis": "Supuesto externo de la demostración sintética",
        "relative_effect": "0.30",
        "relative_effect_basis": "Supuesto de intervención; no aprendido por el clasificador",
        "value_per_case": "25000" if kind == "revenue" else "18000",
        "action_cost_per_case": "1000" if kind == "revenue" else "700",
        "fixed_cost": "2500",
        "value_basis": "Margen o costo evitable constante, sólo para escenario sintético",
    }


def _add_event(
    rows: list[dict], entity: str, moment: pd.Timestamp, stage: str, attributes: dict
) -> None:
    rows.append({"entity": entity, "moment": moment.isoformat(), "stage": stage, **attributes})


def generate(output: Path, *, seed: int = SEED) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "goal_settings-value-demo.v1",
        "seed": seed,
        "synthetic": True,
        "capabilities": {},
    }
    for index, kind in enumerate(("revenue", "savings")):
        rng = random.Random(seed + index)
        folder = output / kind
        folder.mkdir(parents=True, exist_ok=True)
        recipe = _recipe(kind)
        target = recipe["outcome"][0]["output_column"]
        availability = recipe["label_available_at_output_column"]
        files = {}
        group_specs = (
            (
                "initial",
                120,
                "2026-01-01",
                _coverage("2026-01-01T00:00:00Z", "2026-05-31T00:00:00Z"),
            ),
            ("score", 40, "2026-08-15", None),
            (
                "update",
                120,
                "2026-05-01",
                _coverage("2026-05-01T00:00:00Z", "2026-09-30T00:00:00Z"),
            ),
            (
                "compare",
                60,
                "2026-09-15",
                _coverage("2026-09-15T00:00:00Z", "2026-10-20T00:00:00Z"),
            ),
        )
        group_data = {}
        for group, count, start, coverage in group_specs:
            events, outcomes = _events(kind, group, count, start, rng)
            group_data[group] = (events, outcomes, coverage)
        for group in ("initial", "update"):
            events, _outcomes, _coverage_decl = group_data[group]
            path = folder / f"{group}_events.csv"
            path.write_bytes(events.to_csv(index=False, lineterminator="\n").encode())
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        score_events, score_outcomes, _ = group_data["score"]
        scored = build_snapshot(score_events, recipe)
        feature_columns = scored.summary.feature_names
        scoring = scored.dataframe[["entity", *feature_columns]]
        path = folder / "new_cases.csv"
        path.write_bytes(scoring.to_csv(index=False, lineterminator="\n").encode())
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        observations = pd.DataFrame(
            [
                {"entity": entity, target: positive, "observed_at": "2026-09-30T00:00:00Z"}
                for entity, positive in score_outcomes.items()
            ]
        )
        path = folder / "action_outcomes.csv"
        path.write_bytes(observations.to_csv(index=False, lineterminator="\n").encode())
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        compare_events, _compare_outcomes, compare_coverage = group_data["compare"]
        compared = build_snapshot(compare_events, recipe, coverage=compare_coverage)
        if compared.readiness.pending_rows:
            raise ValueError(f"La cohorte {kind} no está madura")
        comparison = compared.training_frame()[["entity", *feature_columns, target, availability]]
        path = folder / "comparison_cohort.csv"
        path.write_bytes(comparison.to_csv(index=False, lineterminator="\n").encode())
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["capabilities"][kind] = {
            "value_kind": kind,
            "question": "¿A quién asistir para aumentar primera compra?"
            if kind == "revenue"
            else "¿Qué pedido verificar antes de prometer entrega?",
            "target": target,
            "availability_column": availability,
            "outcome_window": {"kind": "fixed_days", "days": 14}
            if kind == "revenue"
            else {"kind": "deadline_column", "column": availability},
            "recipe": recipe,
            "coverage": {group: group_data[group][2] for group in ("initial", "update", "compare")},
            "economic_scenario": _economic(kind),
            "files_sha256": files,
            "limits": [
                "La efectividad y los valores económicos son supuestos; "
                "no hay impacto causal real.",
                "La fuente es sintética y la cronología fue generada con semilla fija.",
                "Los casos de scoring y actualización son conjuntos de IDs distintos.",
            ],
        }
        manifest["capabilities"][kind]["counts"] = {
            group: len(group_data[group][1]) for group in group_data
        }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    manifest = generate(args.output, seed=args.seed)
    print(f"Generadas {len(manifest['capabilities'])} capacidades sintéticas en {args.output}")


if __name__ == "__main__":
    main()

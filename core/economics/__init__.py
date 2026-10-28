"""Cálculos económicos declarativos, sin proveedores ni efectos externos."""

from core.economics.scenarios import (
    EconomicCalculationError,
    evaluate_scenario,
    evaluate_scenario_version,
)

__all__ = ["EconomicCalculationError", "evaluate_scenario", "evaluate_scenario_version"]

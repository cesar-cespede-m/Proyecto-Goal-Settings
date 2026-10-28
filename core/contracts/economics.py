"""MV-05: supuestos económicos explícitos; los scores ML no son entradas monetarias."""

from decimal import Context, Decimal, localcontext
from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator, model_validator

from core.domain.context import StrictModel


class EconomicScenarioV1(StrictModel):
    schema_version: Literal["economic-scenario.v1"] = "economic-scenario.v1"
    scenario_kind: Literal["synthetic_demo"] = "synthetic_demo"
    adverse_event: Literal[True] = True
    risk_rate_scope: Literal["selected_cases"] = "selected_cases"
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    currency_minor_units: int | None = Field(default=None, ge=0, le=4, strict=True)
    period: str | None = Field(default=None, min_length=1, max_length=100)
    planned_cases: int | None = Field(default=None, ge=0, le=1_000_000_000, strict=True)
    capacity_cases: int | None = Field(default=None, ge=0, le=1_000_000_000, strict=True)
    external_risk_rate: Decimal | None = Field(
        default=None, ge=0, le=1, max_digits=9, decimal_places=8, allow_inf_nan=False
    )
    risk_rate_basis: str | None = Field(default=None, min_length=1, max_length=2000)
    relative_effect: Decimal | None = Field(
        default=None, ge=0, le=1, max_digits=9, decimal_places=8, allow_inf_nan=False
    )
    relative_effect_basis: str | None = Field(default=None, min_length=1, max_length=2000)
    value_per_case: Decimal | None = Field(
        default=None, ge=0, le=10**12, max_digits=19, decimal_places=6, allow_inf_nan=False
    )
    action_cost_per_case: Decimal | None = Field(
        default=None, ge=0, le=10**12, max_digits=19, decimal_places=6, allow_inf_nan=False
    )
    fixed_cost: Decimal | None = Field(
        default=None, ge=0, le=10**12, max_digits=19, decimal_places=6, allow_inf_nan=False
    )
    value_basis: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="before")
    @classmethod
    def empty_values_are_missing(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: None if isinstance(item, str) and not item.strip() else item
                for key, item in value.items()
            }
        return value

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator(
        "external_risk_rate",
        "relative_effect",
        "value_per_case",
        "action_cost_per_case",
        "fixed_cost",
        mode="before",
    )
    @classmethod
    def require_exact_decimal(cls, value: Any) -> Any:
        if isinstance(value, float | bool):
            raise ValueError("Usa texto decimal o Decimal; no float ni booleanos")
        return value

    @field_serializer(
        "external_risk_rate",
        "relative_effect",
        "value_per_case",
        "action_cost_per_case",
        "fixed_cost",
        when_used="json",
    )
    def serialize_decimal(self, value: Decimal | None) -> str | None:
        with localcontext(Context(prec=50)):
            return (
                None if value is None else ("0" if value == 0 else format(value.normalize(), "f"))
            )

    def missing_fields(self) -> list[str]:
        fields = (
            "currency",
            "currency_minor_units",
            "period",
            "planned_cases",
            "external_risk_rate",
            "risk_rate_basis",
            "relative_effect",
            "relative_effect_basis",
            "value_per_case",
            "action_cost_per_case",
            "fixed_cost",
            "value_basis",
        )
        return [field for field in fields if getattr(self, field) is None]


class EconomicEvaluationV1(StrictModel):
    schema_version: Literal["economic-evaluation.v1"] = "economic-evaluation.v1"
    status: Literal["blocked", "scenario_estimate"]
    evidence_type: Literal["synthetic_scenario"] = "synthetic_scenario"
    formula_version: Literal["external-risk-relative-effect.v1"] = (
        "external-risk-relative-effect.v1"
    )
    scenario_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    missing_fields: list[str]
    currency: str | None = None
    currency_minor_units: int | None = None
    period: str | None = None
    planned_cases: int | None = None
    selected_cases: int | None = None
    expected_avoided_events: Decimal | None = None
    exposure_value: Decimal | None = None
    gross_value: Decimal | None = None
    action_cost: Decimal | None = None
    fixed_cost_applied: Decimal | None = None
    total_cost: Decimal | None = None
    net_value: Decimal | None = None
    warnings: list[str]

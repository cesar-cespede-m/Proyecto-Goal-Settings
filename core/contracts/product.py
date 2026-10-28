"""MV-04: identidad de negocio por encima de proyectos técnicos aislados."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from core.contracts.platform import StrictPlatformModel


class _ProductModel(StrictPlatformModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    id: str = Field(min_length=1, max_length=100)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("La fecha debe incluir zona horaria")
        return value


class ProductV1(_ProductModel):
    schema_version: Literal["product.v1"] = "product.v1"
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    stage: Literal["building", "operating"]
    owner_label: str | None = Field(default=None, min_length=1, max_length=200)
    status: Literal["active", "archived"] = "active"


class CapabilityV1(_ProductModel):
    schema_version: Literal["capability.v1"] = "capability.v1"
    product_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=4000)
    value_kind: Literal["revenue", "savings"]
    decision_unit: str = Field(min_length=1, max_length=200)
    status: Literal["active", "archived"] = "active"


class CapabilityCycleV1(_ProductModel):
    schema_version: Literal["capability-cycle.v1"] = "capability-cycle.v1"
    capability_id: str = Field(min_length=1, max_length=100)
    project_id: str = Field(min_length=1, max_length=100)
    cycle_no: int = Field(ge=1)
    origin: Literal["initial", "update"]
    source_cycle_id: str | None = Field(default=None, min_length=1, max_length=100)
    predictive_spec_id: str | None = Field(default=None, min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def origin_matches_source(self) -> CapabilityCycleV1:
        if (self.origin == "initial") != (self.source_cycle_id is None):
            raise ValueError("Un ciclo de actualización debe identificar su ciclo de origen")
        return self

"""api.schemas -- pydantic request/response models."""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    optimizers: int
    problem_kinds: list[str] = ["qubo", "uc"]


class ParamSchema(BaseModel):
    name: str
    type: Literal["integer", "number", "string", "boolean"]
    default: Any = None
    nullable: bool = False


class OptimizerInfo(BaseModel):
    name: str
    kinds: list[str]
    available: bool
    docstring: str = ""
    params: list[ParamSchema] = Field(default_factory=list)


class OptimizersResponse(BaseModel):
    optimizers: list[OptimizerInfo]


class SelectionRequest(BaseModel):
    """A drawn selection. bbox is frontend order [west, south, east, north]."""
    bbox: list[float] = Field(..., min_length=4, max_length=4)


class NetworkResponse(BaseModel):
    network_id: str
    geojson: dict
    n_bus: int
    n_line: int
    source: str = "openinframap"


class OptimizeRequest(BaseModel):
    network_id: str
    problem_kind: Literal["qubo", "uc"] = "qubo"
    problem_params: dict[str, Any] = Field(default_factory=dict)
    optimizer: str
    solver_params: dict[str, Any] = Field(default_factory=dict)


class OptimizeAccepted(BaseModel):
    job_id: str

"""
api.main -- FastAPI app.

    uvicorn api.main:app --reload --port 8000

Serve the static page separately (`python -m http.server` at repo root, open
/web/opengridmap_source.html). The page works with this server absent; when it is up the page
gains the optimizer panel, live network-from-selection, and the topology view.
"""
from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import CORS_ORIGINS
from .optimizers_api import list_optimizers
from .schemas import HealthResponse, OptimizersResponse
from .routers import network, optimize

app = FastAPI(title="QPGrid API", version="0.1.0",
              description="Local bridge: web map <-> algos/ optimizers <-> OpenInfraMap.")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"],
                   allow_headers=["*"])

app.include_router(network.router)
app.include_router(optimize.router)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(optimizers=len(list_optimizers()))


@app.get("/optimizers", response_model=OptimizersResponse)
def optimizers(kind: str | None = Query(default=None, description="filter: qubo | uc | milp")):
    return OptimizersResponse(optimizers=list_optimizers(kind))

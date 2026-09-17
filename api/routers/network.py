"""api.routers.network -- build a pandapower net from a drawn map selection."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from ..network_builder import build_network_from_box
from ..jobs import register_network
from ..schemas import SelectionRequest, NetworkResponse

router = APIRouter(prefix="/network", tags=["network"])


@router.post("/from-selection", response_model=NetworkResponse)
async def from_selection(req: SelectionRequest):
    try:
        net, geojson = await run_in_threadpool(build_network_from_box, req.bbox)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:                                    # Overpass down, etc.
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    nid = register_network(net)
    n_bus = sum(1 for f in geojson["features"] if f["properties"].get("kind") == "bus")
    n_line = sum(1 for f in geojson["features"] if f["properties"].get("kind") == "line")
    return NetworkResponse(network_id=nid, geojson=geojson, n_bus=n_bus, n_line=n_line)

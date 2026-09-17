"""
api.routers.optimize -- submit an optimization job, stream its progress over SSE.

Optimizer.solve() has no progress callback we can hook (QAOA/VQE keep an internal one, unexposed),
so the stream reports job *status* + heartbeats, then a single terminal event carrying the
serialised Result and, for QUBO, the partitioned GeoJSON. The convergence chart the frontend draws
is a post-hoc replay of Result.trace, not live iteration push. See docs/09-WEB-OVERHAUL.md.
"""
from __future__ import annotations
import asyncio
import json

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from ..jobs import JOBS, create_job, run_job
from ..schemas import OptimizeRequest, OptimizeAccepted

router = APIRouter(prefix="/optimize", tags=["optimize"])


@router.post("", response_model=OptimizeAccepted)
async def submit(req: OptimizeRequest):
    job_id = create_job(req.model_dump())
    asyncio.create_task(run_in_threadpool(run_job, job_id))
    return OptimizeAccepted(job_id=job_id)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/{job_id}/stream")
async def stream(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="unknown job_id")

    async def gen():
        last_status = None
        ticks = 0
        while True:
            job = JOBS[job_id]
            status = job["status"]
            if status != last_status:
                yield _sse("status", {"status": status})
                last_status = status
            if status == "done":
                yield _sse("done", job["result"])
                return
            if status == "error":
                yield _sse("error", job["error"])
                return
            ticks += 1
            if ticks % 20 == 0:                     # ~10 s at 0.5 s cadence
                yield ": heartbeat\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

"""HTTP router of the Hugging Face -> GGUF conversion feature (all routes need the admin key)."""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Depends, HTTPException

from gpupool.converter.models import (
    ClusterVram,
    ConvertError,
    ConvertJob,
    ConvertRequest,
    InspectResult,
    SourceSpec,
)


def _http(e: ConvertError) -> HTTPException:
    return HTTPException(status_code=e.status, detail=e.message)


def make_convert_router(manager, admin_dep, cluster_vram: Callable[[], ClusterVram],
                        inspect: Callable[[SourceSpec], Awaitable[InspectResult]]) -> APIRouter:
    r = APIRouter(dependencies=[Depends(admin_dep)])

    @r.get("/api/convert/options")
    def options() -> dict:
        from gpupool.converter import quant  # lazy: the router must import without the converter
        problem = manager.available()
        return {"available": problem is None, "problem": problem,
                "quant_options": [o.model_dump() for o in quant.QUANT_OPTIONS],
                "cluster": cluster_vram().model_dump()}

    @r.post("/api/convert/inspect")
    async def inspect_source(spec: SourceSpec) -> InspectResult:
        try:
            return await inspect(spec)
        except ConvertError as e:
            raise _http(e) from e

    @r.post("/api/convert")
    async def submit(req: ConvertRequest) -> ConvertJob:
        try:
            return await manager.submit(req)
        except ConvertError as e:
            raise _http(e) from e

    @r.get("/api/convert")
    def list_jobs() -> list[ConvertJob]:
        return manager.list()

    @r.get("/api/convert/{job_id}")
    def get_job(job_id: str) -> ConvertJob:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, f"no conversion job {job_id}")
        return job

    @r.post("/api/convert/{job_id}/cancel")
    async def cancel(job_id: str) -> ConvertJob:
        try:
            return await manager.cancel(job_id)
        except ConvertError as e:
            raise _http(e) from e

    @r.post("/api/convert/{job_id}/retry")
    async def retry(job_id: str) -> ConvertJob:
        try:
            return await manager.retry(job_id)
        except ConvertError as e:
            raise _http(e) from e

    @r.post("/api/convert/{job_id}/accept")
    async def accept(job_id: str) -> ConvertJob:
        try:
            return await manager.accept(job_id)
        except ConvertError as e:
            raise _http(e) from e

    @r.delete("/api/convert/{job_id}")
    async def delete(job_id: str) -> dict:
        try:
            await manager.delete(job_id)
        except ConvertError as e:
            raise _http(e) from e
        return {"ok": True}

    return r

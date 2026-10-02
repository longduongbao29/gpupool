"""HTTP routers for the model library and for serving library files to head nodes."""
from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, model_validator

from gpupool.coordinator.library import Library, LibraryError


class AddLibraryBody(BaseModel):
    hf_repo: str | None = None
    hf_file: str | None = None
    path: str | None = None

    @model_validator(mode="after")
    def _one_form(self):
        hf = self.hf_repo is not None or self.hf_file is not None
        if hf and self.path is not None:
            raise ValueError("give either hf_repo+hf_file or path, not both")
        if hf and not (self.hf_repo and self.hf_file):
            raise ValueError("hf_repo and hf_file are both required")
        if not hf and self.path is None:
            raise ValueError("give hf_repo+hf_file or path")
        return self


def _http(e: LibraryError) -> HTTPException:
    return HTTPException(status_code=e.status, detail=e.message)


def make_library_router(library: Library, admin_dep, in_use: Callable[[str], bool]) -> APIRouter:
    r = APIRouter(dependencies=[Depends(admin_dep)])

    @r.get("/api/hf/files")
    async def hf_files(repo: str):
        try:
            return await library.hf_files(repo)
        except LibraryError as e:
            raise _http(e) from e

    @r.get("/api/library")
    def list_library():
        return [i.model_dump() for i in library.list()]

    @r.post("/api/library")
    async def add_library(body: AddLibraryBody):
        try:
            if body.path is not None:
                item = library.add_path(body.path)
            else:
                item = await library.add_hf(body.hf_repo, body.hf_file)  # type: ignore[arg-type]
        except LibraryError as e:
            raise _http(e) from e
        return item.model_dump()

    @r.delete("/api/library/{name}")
    def delete_library(name: str):
        try:
            library.delete(name, in_use)
        except LibraryError as e:
            raise _http(e) from e
        return {"ok": True}

    return r


def make_files_router(library: Library, cluster_dep) -> APIRouter:
    r = APIRouter(dependencies=[Depends(cluster_dep)])

    @r.get("/files/{name}")
    def get_file(name: str):
        path = library.resolve(name)  # lookup by item name only: no path joining
        if path is None:
            raise HTTPException(status_code=404, detail="no such model file")
        return FileResponse(path, filename=name, media_type="application/octet-stream")

    return r

"""FastAPI application: JSON API plus the built frontend, on one port."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.api import router
from server.jobs import store

_DEFAULT_WEB_DIRS = [
    Path(os.environ.get("A2SB_WEB_DIR", "")) if os.environ.get("A2SB_WEB_DIR") else None,
    Path("/app/web"),
    Path(__file__).resolve().parent.parent / "web" / "dist",
]


def _find_web_dir() -> Path | None:
    for candidate in _DEFAULT_WEB_DIRS:
        if candidate and (candidate / "index.html").is_file():
            return candidate
    return None


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.start()
    try:
        yield
    finally:
        store.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(title="A2SB Restoration", version="2.0.0", lifespan=lifespan)
    # The spectrogram payload is a few hundred KB of base64; everything else is
    # small enough that the threshold keeps compression off the hot path.
    app.add_middleware(GZipMiddleware, minimum_size=8192)
    app.include_router(router)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "activeJobId": store.active_job_id()}

    web_dir = _find_web_dir()
    if web_dir is None:
        @app.get("/")
        def missing_frontend() -> JSONResponse:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": (
                        "Frontend bundle not found. Run `npm ci && npm run build` in web/, "
                        "or set A2SB_WEB_DIR to a directory containing index.html."
                    )
                },
            )
        return app

    index_file = web_dir / "index.html"
    app.mount("/assets", StaticFiles(directory=web_dir / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str) -> FileResponse:
        candidate = (web_dir / full_path).resolve()
        if full_path and web_dir.resolve() in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index_file)

    return app


app = create_app()

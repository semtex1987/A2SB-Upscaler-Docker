"""FastAPI application: JSON API, the built frontend, and the TensorBoard proxy."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from server.api import router
from server.config import TENSORBOARD_PATH_PREFIX
from server.jobs import store
from server.tensorboard import manager as tensorboard

_DEFAULT_WEB_DIRS = [
    Path(os.environ.get("A2SB_WEB_DIR", "")) if os.environ.get("A2SB_WEB_DIR") else None,
    Path("/app/web"),
    Path(__file__).resolve().parent.parent / "web" / "dist",
]

#: Per RFC 7230 these describe a single hop and must not be forwarded.
_HOP_BY_HOP = frozenset({
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
    # Dropped so the app's own GZipMiddleware owns the response framing.
    "content-encoding",
    "content-length",
})

_PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]


def _find_web_dir() -> Path | None:
    for candidate in _DEFAULT_WEB_DIRS:
        if candidate and (candidate / "index.html").is_file():
            return candidate
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.start()
    app.state.proxy_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=None))
    try:
        yield
    finally:
        await app.state.proxy_client.aclose()
        # Stopped before the job store so a TensorBoard holding open file handles
        # on the training tree goes away first.
        tensorboard.stop()
        store.shutdown()


def _register_tensorboard_proxy(app: FastAPI) -> None:
    """Forward `/tensorboard/*` to the loopback TensorBoard.

    Deployments publish only the app's port, so TensorBoard is never reachable
    directly. It runs under a matching `--path_prefix`, which makes this a
    straight path-preserving forward with no URL rewriting.
    """

    @app.api_route(TENSORBOARD_PATH_PREFIX, methods=["GET"], include_in_schema=False)
    def tensorboard_root() -> RedirectResponse:
        # TensorBoard resolves its assets relative to the current path, so
        # without the trailing slash every one of them 404s.
        return RedirectResponse(url=f"{TENSORBOARD_PATH_PREFIX}/", status_code=307)

    @app.api_route(
        TENSORBOARD_PATH_PREFIX + "/{path:path}",
        methods=_PROXY_METHODS,
        include_in_schema=False,
    )
    async def tensorboard_proxy(request: Request, path: str):
        status = tensorboard.status()
        if not status.running:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": (
                        "TensorBoard is not running. Start it from the Train tab, "
                        "or POST /api/training/tensorboard/start."
                    )
                },
            )

        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _HOP_BY_HOP and key.lower() != "host"
        }
        # Identity rather than simply dropping the client's accept-encoding:
        # httpx substitutes its own default when the header is absent, and
        # TensorBoard would then gzip a body whose content-encoding this proxy
        # strips, handing the browser undecodable bytes. Compression happens
        # once, in GZipMiddleware, on the way back out.
        headers["accept-encoding"] = "identity"

        client: httpx.AsyncClient = request.app.state.proxy_client
        upstream_request = client.build_request(
            request.method,
            f"{tensorboard.upstream_base}/{path}",
            headers=headers,
            params=request.query_params,
            content=await request.body(),
        )

        try:
            upstream = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            return JSONResponse(
                status_code=502,
                content={"detail": f"Could not reach TensorBoard: {exc}"},
            )

        async def body():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            body(),
            status_code=upstream.status_code,
            headers={
                key: value
                for key, value in upstream.headers.items()
                if key.lower() not in _HOP_BY_HOP
            },
            media_type=upstream.headers.get("content-type"),
        )


def create_app() -> FastAPI:
    app = FastAPI(title="A2SB Restoration", version="2.0.0", lifespan=lifespan)
    # The spectrogram payload is a few hundred KB of base64; everything else is
    # small enough that the threshold keeps compression off the hot path.
    app.add_middleware(GZipMiddleware, minimum_size=8192)
    app.include_router(router)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "activeJobId": store.active_job_id()}

    # Registered before the SPA catch-all below, which would otherwise swallow
    # every TensorBoard path and hand back index.html.
    _register_tensorboard_proxy(app)

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

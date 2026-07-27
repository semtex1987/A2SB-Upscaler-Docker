"""Entrypoint for the A2SB restoration service.

The application itself lives in `server/`; this module only starts the server so
the container CMD stays `python3 app.py`.
"""
from __future__ import annotations

import os

import uvicorn

from server.main import app  # noqa: F401  - imported so `app.py:app` also works


def main() -> None:
    uvicorn.run(
        "server.main:app",
        host=os.environ.get("A2SB_HOST", "0.0.0.0"),
        port=int(os.environ.get("A2SB_PORT", "7860")),
        log_level=os.environ.get("A2SB_LOG_LEVEL", "info"),
        # A single worker is required: the job store keeps queue state in
        # process memory, and there is one GPU to schedule onto.
        workers=1,
    )


if __name__ == "__main__":
    main()

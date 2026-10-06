"""Entrypoint checkpoint download: cache hits, atomic writes, failed wget."""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ENTRYPOINT = Path(__file__).resolve().parents[1] / "entrypoint.sh"


def _download_ckpt_function() -> str:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    start = text.index("download_ckpt() {")
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def _run_download(tmp_path: Path, *, dest_bytes: bytes | None, wget_ok: bool) -> subprocess.CompletedProcess:
    dest = tmp_path / "A2SB_twosplit_0.0_0.5_release.ckpt"
    if dest_bytes is not None:
        dest.write_bytes(dest_bytes)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wget = bin_dir / "wget"
    if wget_ok:
        wget.write_text(
            "#!/bin/bash\n"
            "dest=\"\"\n"
            "while [[ $# -gt 0 ]]; do\n"
            "  if [[ \"$1\" == \"-O\" ]]; then dest=\"$2\"; shift 2; continue; fi\n"
            "  shift\n"
            "done\n"
            "printf 'fresh' > \"$dest\"\n",
            encoding="utf-8",
        )
    else:
        wget.write_text("#!/bin/bash\necho wget-failed >&2\nexit 1\n", encoding="utf-8")
    wget.chmod(wget.stat().st_mode | stat.S_IXUSR)

    script = tmp_path / "run.sh"
    script.write_text(
        "#!/bin/bash\nset -e\n" + _download_ckpt_function() + "\n"
        f'download_ckpt "{dest}" "https://example.invalid/ckpt"\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    return subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    ), dest


def test_existing_checkpoint_is_not_redownloaded(tmp_path: Path) -> None:
    result, dest = _run_download(tmp_path, dest_bytes=b"cached-weights", wget_ok=True)
    assert result.returncode == 0, result.stderr
    assert dest.read_bytes() == b"cached-weights"
    assert "using cached" in result.stdout


def test_missing_checkpoint_is_downloaded_atomically(tmp_path: Path) -> None:
    result, dest = _run_download(tmp_path, dest_bytes=None, wget_ok=True)
    assert result.returncode == 0, result.stderr
    assert dest.read_text(encoding="utf-8") == "fresh"
    assert not list(tmp_path.glob("*.partial*"))


def test_failed_download_without_cache_is_an_error(tmp_path: Path) -> None:
    result, dest = _run_download(tmp_path, dest_bytes=None, wget_ok=False)
    assert result.returncode != 0
    assert not dest.exists()


def test_failed_download_keeps_existing_file(tmp_path: Path) -> None:
    result, dest = _run_download(tmp_path, dest_bytes=b"cached-weights", wget_ok=False)
    assert result.returncode == 0, result.stderr
    assert dest.read_bytes() == b"cached-weights"

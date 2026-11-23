"""Plugin-facing F5-TTS entry point.

Interface mirrors tts_edge.py: --out, --textfile or --text.
Starts the persistent f5_server.py on the first call and reuses it afterwards.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent
SERVER = PLUGIN_DIR / "f5_server.py"
HOST = "127.0.0.1"
PORT = int(__import__("os").environ.get("F5_PORT", "8766"))
URL = f"http://{HOST}:{PORT}"


def find_python() -> Path:
    """Interpreter with f5-tts installed: portable venv, legacy venv, then current."""
    override = __import__("os").environ.get("F5_PYTHON", "").strip()
    if override:
        return Path(override).expanduser()
    candidates = [
        PLUGIN_DIR / "runtime" / "venv" / "Scripts" / "python.exe",
        PLUGIN_DIR / "runtime" / "venv" / "bin" / "python",
        PLUGIN_DIR / "engines" / "f5" / "venv" / "Scripts" / "python.exe",
        PLUGIN_DIR / "engines" / "f5" / "venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable)


PYTHON = find_python()


def healthy() -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/health", timeout=1.5) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001
        return False


def start_server() -> None:
    env = dict(
        __import__("os").environ,
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        F5_PORT=str(PORT),
    )
    subprocess.Popen(
        [str(PYTHON), str(SERVER)],
        cwd=str(PLUGIN_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def ensure_server() -> None:
    if healthy():
        return
    start_server()
    deadline = time.time() + 120
    while time.time() < deadline:
        if healthy():
            return
        time.sleep(0.5)
    raise RuntimeError("f5 server did not start in 120 s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--textfile", default="")
    parser.add_argument("--text", default="")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--nfe", type=int, default=32)
    parser.add_argument("--warm", action="store_true", help="only start the worker and exit")
    parser.add_argument("--lang", default="ru")
    parser.add_argument("--rate", default="0")
    parser.add_argument("--voice", default="")
    parser.add_argument("--piper", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--fallback-only", action="store_true")
    args, _unknown = parser.parse_known_args()

    if args.warm:
        ensure_server()
        print("[tts-f5] worker ready", file=sys.stderr)
        return 0

    if not args.out:
        print("[tts-f5] --out is required", file=sys.stderr)
        return 1

    text = args.text
    if not text and args.textfile:
        text = Path(args.textfile).read_text(encoding="utf-8")
    text = text.strip()
    if not text:
        print("[tts-f5] empty text", file=sys.stderr)
        return 1

    ensure_server()
    payload = json.dumps({"text": text, "speed": args.speed, "nfe": args.nfe}).encode("utf-8")

    last_error = None
    for attempt in (1, 2):
        try:
            request = urllib.request.Request(
                f"{URL}/tts",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                audio = response.read()
            break
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt == 2:
                raise
            ensure_server()
            time.sleep(1)
    else:
        raise RuntimeError(f"f5 request failed: {last_error}")

    if not audio:
        print("[tts-f5] empty response", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(audio)
    print(f"[tts-f5] ok: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001
        print(f"[tts-f5] failed: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)

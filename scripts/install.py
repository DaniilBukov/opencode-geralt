#!/usr/bin/env python3
"""Cross-platform installer for the Geralt F5-TTS voice plugin.

Creates runtime/venv, installs torch for the current OS, installs the rest of
the requirements, patches ruaccent, downloads the model, and prints the
opencode.json plugin block with paths for THIS machine.

    python scripts/install.py            # NVIDIA GPU (cu124) on Windows/Linux
    python scripts/install.py --cpu      # CPU-only Linux/Windows
    python scripts/install.py --skip-model

Requires Python 3.10+ (3.12 recommended).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / "runtime" / "venv"
TORCH_VERSION = "2.6.0"
DRY_RUN = False


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def run(cmd: list[object]) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"\n$ {printable}", flush=True)
    if DRY_RUN:
        return
    subprocess.check_call([str(part) for part in cmd])


def install_torch(py: Path, cpu_only: bool) -> None:
    if platform.system() == "Darwin":
        run([py, "-m", "pip", "install", f"torch=={TORCH_VERSION}", f"torchaudio=={TORCH_VERSION}"])
        return
    index = "https://download.pytorch.org/whl/cpu" if cpu_only else "https://download.pytorch.org/whl/cu124"
    run(
        [
            py,
            "-m",
            "pip",
            "install",
            f"torch=={TORCH_VERSION}",
            f"torchaudio=={TORCH_VERSION}",
            "--index-url",
            index,
        ]
    )


def plugin_block(py: Path) -> dict:
    name = ROOT.name
    return {
        "package": f"./{name}",
        "options": {
            "enabled": True,
            "command": [
                str(py),
                str(ROOT / "tts_f5.py"),
                "--out",
                "{out}",
                "--textfile",
                "{textfile}",
                "--speed",
                "1.0",
                "--nfe",
                "32",
            ],
            "format": "wav",
            "cwd": str(ROOT),
            "lang": "ru",
            "voice": "",
            "rate": 0,
            "maxChars": 1500,
            "maxTotalChars": 6000,
            "events": ["answer", "permission", "question", "error"],
            "mainSessionOnly": True,
            "debug": True,
            "timeoutMs": 300000,
            "playTimeoutMs": 300000,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable, help="interpreter used to create the venv")
    parser.add_argument("--cpu", action="store_true", help="install CPU-only torch")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without installing anything")
    args = parser.parse_args()

    global DRY_RUN
    DRY_RUN = args.dry_run

    try:
        version = subprocess.check_output(
            [args.python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        version = ""
    if version:
        major, minor = (int(part) for part in version.split("."))
        if (major, minor) < (3, 10) or (major, minor) > (3, 13):
            print(f"warning: Python {version} found; 3.10–3.12 is the safe range")

    py = venv_python()
    if not py.exists():
        run([args.python, "-m", "venv", VENV])
    run([py, "-m", "pip", "install", "--upgrade", "pip", "wheel"])

    if not args.skip_torch:
        install_torch(py, args.cpu)

    run([py, "-m", "pip", "install", "-r", ROOT / "requirements.txt"])
    run([py, ROOT / "scripts" / "patch_ruaccent.py"])

    if not args.skip_model:
        run([py, ROOT / "scripts" / "fetch_model.py"])

    print("\n" + "=" * 70)
    print("Add this block to the `plugins` array in your opencode.json:")
    print("=" * 70)
    print(json.dumps(plugin_block(py), ensure_ascii=False, indent=2))
    print("=" * 70)
    print("Smoke test: " + str(py) + " " + str(ROOT / "tts_f5.py") + " --out test.wav --text \"Проверка связи\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

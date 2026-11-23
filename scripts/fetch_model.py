#!/usr/bin/env python3
"""Download the F5-TTS Russian v2 weights into models/f5.

Files come from https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN
(CC BY-NC 4.0, non-commercial use only).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = ROOT / "models" / "f5"
REPO_ID = "Misha24-10/F5-TTS_RUSSIAN"
FILES = {
    "F5TTS_v1_Base_v2/model_last_inference.safetensors": "model_last_inference.safetensors",
    "F5TTS_v1_Base/vocab.txt": "vocab.txt",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is missing: pip install huggingface_hub")
        return 1

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    for remote, local_name in FILES.items():
        target = dest / local_name
        if target.exists() and not args.force:
            print(f"skip (exists): {target}")
            continue
        print(f"downloading {remote} ...")
        cached = hf_hub_download(repo_id=REPO_ID, filename=remote)
        shutil.copyfile(cached, target)
        print(f"saved: {target}")

    print(f"model directory ready: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

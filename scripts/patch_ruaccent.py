#!/usr/bin/env python3
"""Patch ruaccent for newer transformers/onnx exports.

Some ruaccent exports expect a `token_type_ids` input that the installed
tokenizer does not produce. Without this patch RUAccent.load() raises
"Missing Input: token_type_ids".

The patch is idempotent: running it twice changes nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

ANCHOR = 'inputs = {k: v.astype(np.int64) for k, v in inputs.items()}'
PATCH = ANCHOR + '''
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])'''


def candidate_files() -> list[Path]:
    roots = [Path(p) for p in sys.path if p]
    files = []
    for root in roots:
        path = root / "ruaccent" / "accent_model.py"
        if path.exists():
            files.append(path)
    return files


def main() -> int:
    files = candidate_files()
    if not files:
        print("ruaccent not found; install requirements first", file=sys.stderr)
        return 1

    for path in files:
        source = path.read_text(encoding="utf-8")
        if 'inputs["token_type_ids"] = np.zeros_like' in source:
            print(f"already patched: {path}")
            continue
        count = source.count(ANCHOR)
        if not count:
            print(f"anchor not found, check manually: {path}", file=sys.stderr)
            return 1
        path.write_text(source.replace(ANCHOR, PATCH), encoding="utf-8")
        print(f"patched {count} place(s): {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

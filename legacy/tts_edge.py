from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def log(message: str) -> None:
    print(f"[tts-edge] {message}", file=sys.stderr, flush=True)


async def synth_edge(text: str, out_wav: Path, voice: str, rate: str, pitch: str) -> None:
    import edge_tts
    import soundfile as sf

    fd, mp3_name = tempfile.mkstemp(prefix="opencode-edge-", suffix=".mp3")
    os.close(fd)
    mp3_path = Path(mp3_name)
    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await asyncio.wait_for(communicate.save(str(mp3_path)), timeout=20)
        if not mp3_path.exists() or mp3_path.stat().st_size == 0:
            raise RuntimeError("Edge TTS returned an empty file")
        audio, sample_rate = sf.read(str(mp3_path))
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav), audio, sample_rate)
    finally:
        try:
            mp3_path.unlink()
        except OSError:
            pass


def synth_piper(text: str, out_wav: Path, piper: str, model: str) -> None:
    if not piper or not model:
        raise RuntimeError("Piper fallback is not configured")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [piper, "-m", model, "-f", str(out_wav)],
        input=text.encode("utf-8"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Piper exited with code {result.returncode}: {result.stderr.decode('utf-8', 'replace')[-1000:]}")
    if not out_wav.exists() or out_wav.stat().st_size == 0:
        raise RuntimeError("Piper returned an empty file")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--textfile", default="")
    parser.add_argument("--voice", default="ru-RU-SvetlanaNeural")
    parser.add_argument("--rate", default="+0%")
    parser.add_argument("--pitch", default="+0Hz")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--piper", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--fallback-only", action="store_true")
    args = parser.parse_args()

    text = (
        Path(args.textfile).read_text(encoding="utf-8")
        if args.textfile
        else sys.stdin.read()
    ).strip()
    if not text:
        log("empty text")
        return 1

    out_wav = Path(args.out)

    if not args.fallback_only:
        attempts = max(1, args.retries)
        for attempt in range(1, attempts + 1):
            try:
                asyncio.run(synth_edge(text, out_wav, args.voice, args.rate, args.pitch))
                log(f"edge ok (attempt {attempt}): {out_wav}")
                return 0
            except Exception as error:  # noqa: BLE001
                log(f"edge failed (attempt {attempt}/{attempts}): {type(error).__name__}: {error}")
                if attempt < attempts:
                    import time

                    time.sleep(1.5)
        if args.no_fallback:
            return 1

    try:
        synth_piper(text, out_wav, args.piper, args.model)
        log(f"piper fallback ok: {out_wav}")
        return 0
    except Exception as error:  # noqa: BLE001
        log(f"piper fallback failed: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

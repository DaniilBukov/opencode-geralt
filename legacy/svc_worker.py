from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

# Keep stdout clean for the line-based JSON protocol. Everything else,
# including library prints and progress bars, goes to stderr.
PROTOCOL = sys.stdout
sys.stdout = sys.stderr


def send(payload: dict) -> None:
    PROTOCOL.write(json.dumps(payload, ensure_ascii=True) + "\n")
    PROTOCOL.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cluster-model", default="")
    parser.add_argument("--cluster-ratio", type=float, default=0.0)
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--half", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Import heavy modules only after stdout has been redirected.
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from so_vits_svc_fork.inference.core import Svc

    model = Path(args.model)
    config = Path(args.config)
    if not model.exists():
        raise FileNotFoundError(model)
    if not config.exists():
        raise FileNotFoundError(config)

    print(f"[svc-worker] loading model {model}", file=sys.stderr, flush=True)
    svc = Svc(
        net_g_path=str(model),
        config_path=str(config),
        cluster_model_path=args.cluster_model or None,
        device=args.device,
        half=args.half,
    )
    speaker = args.speaker or next(iter(svc.spk2id))
    print(f"[svc-worker] ready; speaker={speaker}; device={svc.device}", file=sys.stderr, flush=True)
    send({"ready": True, "speaker": speaker, "sampleRate": svc.target_sample})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            input_path = Path(request["input"])
            output_path = Path(request["output"])
            if not input_path.exists():
                raise FileNotFoundError(input_path)

            audio, _ = librosa.load(str(input_path), sr=svc.target_sample)
            result = svc.infer_silence(
                audio.astype(np.float32),
                speaker=request.get("speaker", speaker),
                transpose=int(request.get("transpose", 0)),
                auto_predict_f0=bool(request.get("autoPredictF0", False)),
                cluster_infer_ratio=float(request.get("clusterInferRatio", args.cluster_ratio)),
                noise_scale=float(request.get("noiseScale", 0.4)),
                f0_method=request.get("f0Method", "dio"),
                db_thresh=int(request.get("dbThresh", -40)),
                pad_seconds=float(request.get("padSeconds", 0.5)),
                chunk_seconds=float(request.get("chunkSeconds", 0.5)),
                absolute_thresh=bool(request.get("absoluteThresh", False)),
                max_chunk_seconds=float(request.get("maxChunkSeconds", 40)),
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), result, svc.target_sample, subtype="PCM_16")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            send({"id": request_id, "ok": True})
        except Exception as error:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            send({
                "id": request.get("id") if isinstance(request, dict) else None,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            })


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        send({"ready": False, "error": f"{type(error).__name__}: {error}"})
        sys.exit(1)

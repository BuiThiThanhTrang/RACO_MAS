from __future__ import annotations

import argparse
import json
from pathlib import Path


def _boolean(value: str) -> bool:
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolated faster-whisper worker for GAIA attachments"
    )
    parser.add_argument("--file", required=True)
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--vad-filter", type=_boolean, default=True)
    parser.add_argument("--download-root")
    args = parser.parse_args()

    source = Path(args.file).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    from faster_whisper import WhisperModel

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        download_root=args.download_root,
    )
    segments, info = model.transcribe(
        str(source), beam_size=args.beam_size, vad_filter=args.vad_filter
    )
    payload = {
        "language": getattr(info, "language", None),
        "duration": float(getattr(info, "duration", 0.0) or 0.0),
        "segments": [
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": str(segment.text or ""),
            }
            for segment in segments
        ],
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()

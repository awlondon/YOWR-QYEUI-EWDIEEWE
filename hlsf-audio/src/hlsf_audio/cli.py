from __future__ import annotations

import argparse
import json
from pathlib import Path

from hlsf_audio.api import ingest_and_extract_wav


def main() -> int:
    p = argparse.ArgumentParser(prog="hlsf-audio", description="HLSF Audio IR bootstrap CLI")

    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Ingest WAV and run deterministic extractor")
    ing.add_argument("wav", type=str, help="Path to .wav file")
    ing.add_argument("--out", type=str, default="ir.json", help="Output IR JSON file path")
    ing.add_argument(
        "--blob-mode",
        type=str,
        choices=["memory", "fs"],
        default="fs",
        help="Blob storage mode",
    )
    ing.add_argument("--blob-root", type=str, default="blobs", help="Blob directory (fs mode)")

    args = p.parse_args()

    if args.cmd == "ingest":
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ir = ingest_and_extract_wav(
            args.wav,
            blob_mode=args.blob_mode,
            blob_root=args.blob_root,
        )

        out_path.write_text(json.dumps(ir, indent=2), encoding="utf-8")
        print(f"Wrote IR: {out_path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

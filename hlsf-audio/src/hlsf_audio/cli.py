from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import ingest_and_extract_wav


def _cmd_ingest(args: argparse.Namespace) -> int:
    blob_mode = "memory"
    blob_root: str | None = None
    if args.blobs:
        blob_mode = "fs"
        blob_root = args.blobs

    ir_dict = ingest_and_extract_wav(args.path, blob_mode=blob_mode, blob_root=blob_root)

    if args.out in (None, "-"):
        print(json.dumps(ir_dict, indent=2, sort_keys=True))
        return 0

    out_path = Path(args.out)
    if out_path.parent != Path("."):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(ir_dict, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m hlsf_audio.cli")
    subparsers = parser.add_subparsers(dest="command")

    ingest = subparsers.add_parser("ingest", help="Ingest a WAV file and write the IR JSON.")
    ingest.add_argument("path", help="Path to WAV file")
    ingest.add_argument("--out", required=True, help="Output JSON path, or '-' for stdout")
    ingest.add_argument(
        "--blobs",
        help="Directory to store extracted blobs (enables filesystem blob store)",
    )
    ingest.set_defaults(func=_cmd_ingest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

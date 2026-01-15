from __future__ import annotations
from pathlib import Path
import numpy as np
from .core.ingest import load_wav, ingest_audio
from .core.extractor import run_deterministic_extractor, ExtractorConfig
from .ir.delta import apply_delta
from .ir.validate import validate_ir_dict
from .storage.blobstore import BlobStore


def ingest_and_extract_wav(
    path: str | Path,
    blob_mode: str = "memory",
    blob_root: str | None = None,
    cfg: ExtractorConfig | None = None,
) -> dict:
    audio, sr, ch = load_wav(path)
    ir = ingest_audio(audio, sr=sr, channels=ch)

    blob = BlobStore(mode=blob_mode, root=Path(blob_root) if blob_root else None)
    d = run_deterministic_extractor(ir, audio=audio, sr=sr, blob=blob, cfg=cfg or ExtractorConfig())
    ir = apply_delta(ir, d)

    ir_dict = ir.model_dump()
    validate_ir_dict(ir_dict)
    return ir_dict


def ingest_and_extract_array(audio: np.ndarray, sr: int, channels: int = 1, blob_mode: str = "memory") -> dict:
    ir = ingest_audio(audio, sr=sr, channels=channels)
    blob = BlobStore(mode=blob_mode)
    d = run_deterministic_extractor(ir, audio=audio, sr=sr, blob=blob)
    ir = apply_delta(ir, d)
    ir_dict = ir.model_dump()
    validate_ir_dict(ir_dict)
    return ir_dict

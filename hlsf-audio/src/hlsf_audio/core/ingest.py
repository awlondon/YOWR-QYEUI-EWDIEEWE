from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from ..ir.types import IR, IRMeta, IRMetaSource, Timebases, FrameTimebase, Track


@dataclass
class IngestConfig:
    sr: int | None = None  # if provided and WAV differs, raise for now


def _to_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype.kind in ("i", "u"):
        maxv = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / float(maxv)
    else:
        x = x.astype(np.float32)
    return x


def load_wav(path: str | Path, cfg: IngestConfig = IngestConfig()) -> tuple[np.ndarray, int, int]:
    sr, data = wavfile.read(str(path))
    if cfg.sr is not None and sr != cfg.sr:
        raise ValueError(f"Sample rate mismatch: {sr} != {cfg.sr} (resampling not implemented)")
    data = _to_float32(np.asarray(data))
    if data.ndim == 1:
        channels = 1
        mono = data
    else:
        channels = data.shape[1]
        mono = data.mean(axis=1)
    return mono, sr, channels


def ingest_audio(audio: np.ndarray, sr: int, channels: int = 1) -> IR:
    audio = _to_float32(np.asarray(audio).reshape(-1))
    duration_s = float(len(audio) / sr)
    now = datetime.now(timezone.utc).isoformat()

    timebases = Timebases(
        samples={"sr": sr},
        frames=[
            FrameTimebase(name="stft_1024_h256", win=1024, hop=256),
            FrameTimebase(name="stft_4096_h1024", win=4096, hop=1024),
        ],
    )

    ir = IR(
        meta=IRMeta(created_utc=now, source=IRMetaSource(sr=sr, channels=channels, duration_s=duration_s)),
        timebases=timebases,
        tracks=[Track(id="mix", type="mixture")],
        fields={},
        events=[],
        segments=[],
        graph={"nodes": [], "edges": []},
        evidence=[],
    )
    return ir

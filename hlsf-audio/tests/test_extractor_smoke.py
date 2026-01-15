import numpy as np
from hlsf_audio.core.ingest import ingest_audio
from hlsf_audio.core.extractor import run_deterministic_extractor
from hlsf_audio.storage.blobstore import BlobStore
from hlsf_audio.ir.delta import apply_delta
from hlsf_audio.ir.validate import validate_ir_dict


def test_extractor_smoke():
    sr = 48000
    t = np.arange(sr, dtype=np.float32) / sr
    audio = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    audio[10000:10010] += 0.9
    audio[30000:30010] += 0.9

    ir = ingest_audio(audio, sr=sr, channels=1)
    blob = BlobStore(mode="memory")
    d = run_deterministic_extractor(ir, audio=audio, sr=sr, blob=blob)
    ir = apply_delta(ir, d)
    out = ir.model_dump()
    validate_ir_dict(out)

    assert any(k.endswith("/stft/logmag") for k in out["fields"].keys())
    assert len(out["events"]) >= 1
    assert len(out["segments"]) >= 1

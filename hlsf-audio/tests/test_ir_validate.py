import numpy as np
from hlsf_audio.core.ingest import ingest_audio
from hlsf_audio.ir.validate import validate_ir_dict


def test_ingest_validates():
    audio = np.zeros(48000, dtype=np.float32)
    ir = ingest_audio(audio, sr=48000, channels=1)
    validate_ir_dict(ir.model_dump())

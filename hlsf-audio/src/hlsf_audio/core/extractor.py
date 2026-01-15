from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import stft
from ..ir.types import FieldSpec, FieldRef, TimedAtom, Evidence
from ..ir.delta import IRDelta
from ..storage.blobstore import BlobStore


@dataclass
class ExtractorConfig:
    stfts: tuple[tuple[str, int, int], ...] = (
        ("stft_1024_h256", 1024, 256),
        ("stft_4096_h1024", 4096, 1024),
    )
    vad_rms_db_threshold: float = -45.0
    vad_min_on_s: float = 0.10
    vad_min_off_s: float = 0.10
    onset_z_thresh: float = 2.5
    onset_min_gap_s: float = 0.05


def _uid(prefix: str, i: int) -> str:
    return f"{prefix}_{i:06d}"


def run_deterministic_extractor(ir, audio: np.ndarray, sr: int, blob: BlobStore, cfg: ExtractorConfig = ExtractorConfig()) -> IRDelta:
    d = IRDelta()
    event_i = 0
    seg_i = 0
    ev_i = 0

    def add_evidence(kind: str, ref: str, t0: float, t1: float, note: str) -> str:
        nonlocal ev_i
        eid = _uid("evid", ev_i)
        ev_i += 1
        d.add_evidence.append(
            Evidence(id=eid, kind=kind, ref=ref, span={"t0": float(t0), "t1": float(t1)}, note=note)
        )
        return eid

    for (tb_name, win, hop) in cfg.stfts:
        f, t, Zxx = stft(audio, fs=sr, nperseg=win, noverlap=win - hop, boundary=None, padded=False)
        mag = np.abs(Zxx).astype(np.float32)
        logmag = np.log1p(mag)

        store, key = blob.put(f"mix_{tb_name}_stft_logmag", logmag)
        field_key = f"mix/{tb_name}/stft/logmag"
        d.add_fields[field_key] = FieldSpec(
            kind="ndarray",
            shape=list(logmag.shape),
            dtype="f32",
            ref=FieldRef(store=store, key=key),
            timebase=tb_name,
            track="mix",
            desc="log(1+|STFT|)",
        )

        rms = np.sqrt(np.mean(mag * mag, axis=0) + 1e-12).astype(np.float32)
        rms_db = (20.0 * np.log10(rms + 1e-12)).astype(np.float32)

        store2, key2 = blob.put(f"mix_{tb_name}_feat_rms_db", rms_db)
        rms_key = f"mix/{tb_name}/feat/rms_db"
        d.add_fields[rms_key] = FieldSpec(
            kind="ndarray",
            shape=[int(rms_db.shape[0])],
            dtype="f32",
            ref=FieldRef(store=store2, key=key2),
            timebase=tb_name,
            track="mix",
            desc="per-frame RMS in dB (proxy from STFT magnitude)",
        )

        active = rms_db >= cfg.vad_rms_db_threshold

        def emit_segments(mask: np.ndarray):
            nonlocal seg_i
            min_on = int(np.ceil(cfg.vad_min_on_s / (hop / sr)))
            min_off = int(np.ceil(cfg.vad_min_off_s / (hop / sr)))
            del min_off

            start = None
            for i, v in enumerate(mask):
                if v and start is None:
                    start = i
                if (not v or i == len(mask) - 1) and start is not None:
                    end = i if not v else i + 1
                    length = end - start
                    if length >= min_on:
                        t0 = float(t[start]) if start < len(t) else float(start * hop / sr)
                        t1 = float(t[end - 1]) if (end - 1) < len(t) else float(end * hop / sr)
                        evid = add_evidence("frame_range", rms_key, t0, t1, f"VAD on {tb_name} rms_db>=thresh")
                        d.add_segments.append(
                            TimedAtom(
                                id=_uid("seg", seg_i),
                                t0=t0,
                                t1=t1,
                                type="non_silence",
                                confidence=0.6,
                                track="mix",
                                tags=[tb_name],
                                attrs={"rms_db_threshold": cfg.vad_rms_db_threshold},
                                evidence=[evid],
                            )
                        )
                        seg_i += 1
                    start = None

        emit_segments(active)

        if mag.shape[1] >= 2:
            diff = mag[:, 1:] - mag[:, :-1]
            flux = np.sum(np.maximum(diff, 0.0), axis=0).astype(np.float32)
            mu = float(np.mean(flux))
            sd = float(np.std(flux) + 1e-9)
            z = (flux - mu) / sd
            min_gap_frames = int(np.ceil(cfg.onset_min_gap_s / (hop / sr)))
            last = -10**9
            for k, zk in enumerate(z):
                if zk >= cfg.onset_z_thresh and (k - last) >= min_gap_frames:
                    last = k
                    onset_t = float(t[k + 1]) if (k + 1) < len(t) else float((k + 1) * hop / sr)
                    evid = add_evidence(
                        "frame_range",
                        field_key,
                        max(0.0, onset_t - 0.02),
                        onset_t + 0.02,
                        f"spectral flux z>={cfg.onset_z_thresh}",
                    )
                    d.add_events.append(
                        TimedAtom(
                            id=_uid("evt", event_i),
                            t0=onset_t,
                            t1=onset_t,
                            type="onset",
                            confidence=min(1.0, 0.5 + 0.1 * float(zk)),
                            track="mix",
                            tags=[tb_name],
                            attrs={"z": float(zk), "hop": hop, "win": win},
                            evidence=[evid],
                        )
                    )
                    event_i += 1

    return d

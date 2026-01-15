# HLSF Audio (IR v0.1)

This repo boots a stable intermediate representation (IR) for audio:
- Ingest WAV/audio arrays
- Deterministic extraction: multi-res STFT logmag + RMS + simple VAD + basic onset events
- JSON schema validation
- Blobstore for large arrays

## Quickstart
1) python -m venv .venv && source .venv/bin/activate
2) pip install -e .
3) pytest -q

## Example usage (in Python)
- from hlsf_audio.api import ingest_and_extract_wav

## CLI usage
Ingest a WAV file to JSON with optional blob storage:

```
python -m hlsf_audio.cli ingest path.wav --out ir.json --blobs blobs/
```

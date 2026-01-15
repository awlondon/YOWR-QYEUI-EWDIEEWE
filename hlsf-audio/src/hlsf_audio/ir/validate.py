from __future__ import annotations
import json
from pathlib import Path
from jsonschema import Draft202012Validator

_SCHEMA_PATH = Path(__file__).with_name("schema.json")
_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
_VALIDATOR = Draft202012Validator(_SCHEMA)


def validate_ir_dict(ir: dict) -> None:
    errors = sorted(_VALIDATOR.iter_errors(ir), key=lambda e: e.path)
    if errors:
        msg = "\n".join([f"{list(e.path)}: {e.message}" for e in errors[:20]])
        raise ValueError(f"IR schema validation failed:\n{msg}")

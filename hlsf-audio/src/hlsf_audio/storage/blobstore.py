from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Literal, Optional
import numpy as np

StoreKind = Literal["memory", "fs"]


@dataclass
class BlobStore:
    mode: StoreKind = "memory"
    root: Optional[Path] = None
    _mem: Dict[str, np.ndarray] = field(default_factory=dict)

    def put(self, key_hint: str, arr: np.ndarray) -> Tuple[StoreKind, str]:
        key = self._sanitize_key(key_hint)
        if self.mode == "memory":
            self._mem[key] = np.asarray(arr)
            return "memory", key
        if self.root is None:
            raise ValueError("BlobStore in fs mode requires root")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{key}.npy"
        np.save(path, np.asarray(arr), allow_pickle=False)
        return "fs", str(path)

    def get(self, key: str) -> np.ndarray:
        if self.mode == "memory":
            return self._mem[key]
        return np.load(Path(key), allow_pickle=False)

    @staticmethod
    def _sanitize_key(s: str) -> str:
        return "".join(c if (c.isalnum() or c in "._-") else "_" for c in s)[:120]

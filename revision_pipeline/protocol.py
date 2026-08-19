from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Protocol:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Protocol":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "protocol_version",
            "frozen_on",
            "eeg_channels",
            "seeds",
            "noninferiority_margin",
            "models",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"Protocol is missing fields: {missing}")
        if data["eeg_channels"] != ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "C2"]:
            raise ValueError("The confirmatory protocol must use exactly the eight declared EEG channels")
        if len(data["seeds"]) != 5 or len(set(data["seeds"])) != 5:
            raise ValueError("Exactly five distinct ensemble seeds are required")
        margin = float(data["noninferiority_margin"])
        if not 0.0 < margin < 0.5:
            raise ValueError("noninferiority_margin must be expressed as a proportion")
        return cls(data)

    @property
    def seeds(self) -> list[int]:
        return [int(x) for x in self.raw["seeds"]]

    @property
    def digest(self) -> str:
        canonical = json.dumps(self.raw, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def model(self, name: str) -> dict[str, Any]:
        return dict(self.raw["models"][name])

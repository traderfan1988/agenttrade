from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class Zustand(Enum):
    UNBESTIMMT = "UNBESTIMMT"        # Nicht genug Daten für Bewertung
    PASSIERT = "PASSIERT"            # Bedingung eingetreten / Kriterium erfüllt
    NICHT_PASSIERT = "NICHT_PASSIERT"  # Bedingung nicht eingetreten


@dataclass
class Befund:
    label: str
    wert: Any = None
    bestanden: Optional[bool] = None  # None = nicht bewertbar (≠ False)
    zustand: Zustand = Zustand.UNBESTIMMT
    details: str = ""

    def __str__(self) -> str:
        if self.bestanden is None:
            sym = "?"
        elif self.bestanden:
            sym = "✓"
        else:
            sym = "✗"
        if isinstance(self.wert, float):
            wert_str = f"{self.wert:.4f}"
        elif self.wert is None:
            wert_str = "—"
        else:
            wert_str = str(self.wert)
        return f"[{sym}] {self.label}: {wert_str}"

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "wert": self.wert,
            "bestanden": self.bestanden,
            "zustand": self.zustand.value,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Befund":
        d = dict(d)
        d["zustand"] = Zustand(d.get("zustand", "UNBESTIMMT"))
        return cls(**d)


def befund_unbestimmt(label: str, details: str = "") -> Befund:
    """Factory: immer verwenden wenn Daten fehlen. Niemals PASSIERT zurückgeben."""
    return Befund(
        label=label,
        wert=None,
        bestanden=None,
        zustand=Zustand.UNBESTIMMT,
        details=details or "Keine Daten verfügbar",
    )

"""Tests für kern/typen.py – Invarianten aus den Arbeitsregeln."""
import pytest
from kern.typen import Befund, Zustand, befund_unbestimmt


class TestBefundUnbestimmt:
    def test_bestanden_ist_none_nicht_false(self):
        b = befund_unbestimmt("Test")
        assert b.bestanden is None
        assert b.bestanden != False  # noqa: E712  bestanden=None ≠ nein

    def test_zustand_ist_unbestimmt(self):
        b = befund_unbestimmt("Test")
        assert b.zustand == Zustand.UNBESTIMMT

    def test_wert_ist_none(self):
        b = befund_unbestimmt("Test")
        assert b.wert is None

    def test_details_wird_gesetzt(self):
        b = befund_unbestimmt("X", "Mein Grund")
        assert "Mein Grund" in b.details

    def test_details_hat_fallback(self):
        b = befund_unbestimmt("X")
        assert b.details  # nicht leer


class TestBefundStr:
    def test_unbestimmt_zeigt_fragezeichen(self):
        b = befund_unbestimmt("KGV")
        assert "[?]" in str(b)

    def test_bestanden_zeigt_haken(self):
        b = Befund(label="KGV", wert=15.0, bestanden=True)
        assert "[✓]" in str(b)

    def test_nicht_bestanden_zeigt_kreuz(self):
        b = Befund(label="KGV", wert=25.0, bestanden=False)
        assert "[✗]" in str(b)

    def test_float_wert_wird_formatiert(self):
        b = Befund(label="X", wert=0.1234, bestanden=True)
        assert "0.1234" in str(b)

    def test_none_wert_zeigt_strich(self):
        b = Befund(label="X", wert=None, bestanden=None)
        assert "—" in str(b)


class TestBefundSerialisierung:
    def test_as_dict_runde(self):
        original = Befund(
            label="KGV", wert=15.0, bestanden=True,
            zustand=Zustand.PASSIERT, details="ok"
        )
        d = original.as_dict()
        rekonstruiert = Befund.from_dict(d)
        assert rekonstruiert.label == original.label
        assert rekonstruiert.wert == original.wert
        assert rekonstruiert.bestanden == original.bestanden
        assert rekonstruiert.zustand == original.zustand

    def test_zustand_als_string_in_dict(self):
        b = Befund(label="X", zustand=Zustand.PASSIERT)
        d = b.as_dict()
        assert d["zustand"] == "PASSIERT"


class TestArbeitsregeln:
    """Explizite Tests für die Arbeitsregeln aus CLAUDE.md."""

    def test_fehlende_daten_nie_passiert(self):
        b = befund_unbestimmt("Test")
        assert b.zustand != Zustand.PASSIERT, (
            "CLAUDE.md: Fehlende Daten → UNBESTIMMT, niemals PASSIERT"
        )

    def test_bestanden_none_ist_nicht_bewertbar(self):
        b = Befund(label="X", bestanden=None)
        bewertbar = b.bestanden is not None
        assert not bewertbar, (
            "CLAUDE.md: bestanden=None heißt 'nicht bewertbar', nicht 'nein'"
        )

    def test_befund_hat_label(self):
        b = Befund(label="KGV", wert=15.0)
        assert b.label, "CLAUDE.md: Kein Agent gibt nackte Zahl zurück, nur Befund mit Label"

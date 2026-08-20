"""
Tests für Agent 2 – Drawdown-Analyse.

Kernregel: Drawdown gegen Drawdown im selben Fenster.
Fenster ist am 52W-Hoch verankert → kein Rendite-Bias.
"""
from unittest.mock import patch

import pandas as pd
import pytest

from kern.typen import Befund, Zustand
from agenten.agent2_drawdown import (
    SCHWELLEN,
    _drawdown_aktuell,
    _fenster_am_hoch_verankert,
    _max_dd_im_fenster,
    _tage_seit_hoch,
    analyse,
    conviction,
)


def _serie(werte: list) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(werte), freq="B")
    return pd.Series(werte, index=idx, dtype=float)


class TestDrawdownAktuell:
    def test_korrekte_berechnung(self):
        s = _serie([100, 110, 120, 90, 95])
        dd = _drawdown_aktuell(s)
        assert abs(dd - (120 - 95) / 120) < 1e-9

    def test_immer_positiv(self):
        s = _serie([80, 90, 100, 85, 70, 75])
        dd = _drawdown_aktuell(s)
        assert dd >= 0, "Drawdown ist immer ≥ 0"

    def test_am_hoch_gibt_null(self):
        s = _serie([80, 90, 100])
        dd = _drawdown_aktuell(s)
        assert abs(dd) < 1e-9

    def test_leere_serie_gibt_none(self):
        assert _drawdown_aktuell(pd.Series(dtype=float)) is None

    def test_null_preis_gibt_none(self):
        assert _drawdown_aktuell(_serie([0, 0, 0])) is None


class TestFensterAmHochVerankert:
    def test_fenster_startet_am_hoch(self):
        s = _serie([80, 90, 100, 85, 70, 75])
        fenster = _fenster_am_hoch_verankert(s)
        assert float(fenster.iloc[0]) == 100.0

    def test_fenster_endet_am_letzten_wert(self):
        s = _serie([80, 90, 100, 85, 70, 75])
        fenster = _fenster_am_hoch_verankert(s)
        assert float(fenster.iloc[-1]) == 75.0

    def test_laenge_korrekt(self):
        # Max bei Index 2, Serie hat 6 Elemente → Fenster hat 4
        s = _serie([80, 90, 100, 85, 70, 75])
        fenster = _fenster_am_hoch_verankert(s)
        assert len(fenster) == 4

    def test_leere_serie_bleibt_leer(self):
        s = pd.Series(dtype=float)
        fenster = _fenster_am_hoch_verankert(s)
        assert fenster.empty


class TestMaxDDImFenster:
    def test_tiefster_punkt_wird_gefunden(self):
        # Fenster startet bei 100, tiefstes Tief bei 70 → max DD = 30%
        fenster = _serie([100, 90, 70, 80])
        dd = _max_dd_im_fenster(fenster)
        assert abs(dd - 30 / 100) < 1e-9

    def test_zu_kurze_serie_gibt_none(self):
        assert _max_dd_im_fenster(_serie([100])) is None

    def test_immer_positiv(self):
        fenster = _serie([100, 90, 80])
        dd = _max_dd_im_fenster(fenster)
        assert dd >= 0


class TestTagesSeitHoch:
    def test_tage_korrekt_wenn_hoch_am_anfang(self):
        s = _serie([100, 90, 80, 70])  # Hoch am Tag 0
        assert _tage_seit_hoch(s) == 3

    def test_tage_null_wenn_hoch_heute(self):
        s = _serie([80, 90, 100])  # Hoch am letzten Tag
        assert _tage_seit_hoch(s) == 0

    def test_tage_mitte(self):
        s = _serie([80, 100, 90, 85, 80])  # Hoch bei Index 1
        assert _tage_seit_hoch(s) == 3

    def test_leere_serie_gibt_none(self):
        assert _tage_seit_hoch(pd.Series(dtype=float)) is None

    def test_immer_nicht_negativ(self):
        s = _serie([50, 80, 120, 90, 100])
        tage = _tage_seit_hoch(s)
        assert tage >= 0


class TestKernregel_DrawdownGegenDrawdown:
    """Expliziter Test der Kernregel: Drawdown vs. Drawdown, nicht Rendite vs. Drawdown."""

    def test_vergleich_ist_drawdown_nicht_rendite(self):
        # Fenster: Hoch 120, Tief 90, Aktuell 100
        # Drawdown vom Hoch = (120-100)/120 ≈ 16.67%
        # Rendite seit Hoch = (100-120)/120 ≈ -16.67%  (FALSCH - negativer Wert = Bias)
        s = _serie([100, 120, 110, 90, 100])
        fenster = _fenster_am_hoch_verankert(s)
        max_dd = _max_dd_im_fenster(fenster)
        akt_dd = _drawdown_aktuell(s)
        # Beide müssen positiv sein → Drawdown, nicht Rendite
        assert max_dd >= 0, "max_dd muss positiv sein (Drawdown, nicht Rendite)"
        assert akt_dd >= 0, "akt_dd muss positiv sein (Drawdown, nicht Rendite)"

    def test_fenster_bias_durch_verankerung_vermieden(self):
        # Wäre das Fenster NICHT am Hoch verankert und wir würden Renditen messen,
        # wären alle Werte im Fenster ≤ 0 (weil Hoch = Start).
        # Durch Drawdown-Berechnung ist das Ergebnis immer ≥ 0.
        s = _serie([80, 100, 95, 85, 90])  # Hoch bei Index 1 (100)
        fenster = _fenster_am_hoch_verankert(s)
        assert float(fenster.iloc[0]) == 100.0, "Fenster muss am Hoch starten"
        dd = _max_dd_im_fenster(fenster)
        assert dd >= 0, "Drawdown im Fenster ist immer ≥ 0 (kein Rendite-Bias)"


class TestAnalyse:
    def _make_df(self, werte: list) -> pd.DataFrame:
        idx = pd.date_range("2022-01-01", periods=len(werte), freq="B")
        return pd.DataFrame({"Close": werte}, index=idx)

    def test_leere_daten_geben_unbestimmt(self):
        with patch("agenten.agent2_drawdown.yf.download") as mock_dl:
            mock_dl.return_value = pd.DataFrame()
            befunde = analyse("TEST")
        for b in befunde:
            assert b.bestanden is None
            assert b.zustand == Zustand.UNBESTIMMT

    def test_zu_wenig_daten_geben_unbestimmt(self):
        with patch("agenten.agent2_drawdown.yf.download") as mock_dl:
            mock_dl.return_value = self._make_df([100.0] * 10)
            befunde = analyse("TEST")
        for b in befunde:
            assert b.bestanden is None

    def test_gibt_nur_befund_objekte_zurueck(self):
        werte = [100.0] * 300 + [80.0] * 200
        with patch("agenten.agent2_drawdown.yf.download") as mock_dl:
            mock_dl.return_value = self._make_df(werte)
            befunde = analyse("TEST")
        for b in befunde:
            assert isinstance(b, Befund), "kein Agent gibt nackte Zahl zurück"

    def test_exception_gibt_unbestimmt(self):
        with patch("agenten.agent2_drawdown.yf.download") as mock_dl:
            mock_dl.side_effect = RuntimeError("Timeout")
            befunde = analyse("TEST")
        assert all(b.bestanden is None for b in befunde)


class TestConviction:
    def test_alle_bestanden_gibt_100(self):
        befunde = [Befund(label=f"X{i}", bestanden=True) for i in range(3)]
        assert conviction(befunde) == 100.0

    def test_keine_bewertbaren_gibt_0(self):
        befunde = [Befund(label="X", bestanden=None)]
        assert conviction(befunde) == 0.0

    def test_schwellen_kommen_aus_dict(self):
        assert "drawdown_guenstig_min" in SCHWELLEN
        assert "fenster_tage" in SCHWELLEN
        assert "conviction_gewicht" in SCHWELLEN

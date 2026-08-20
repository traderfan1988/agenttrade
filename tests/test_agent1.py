"""Tests für Agent 1 – Bewertung. Mocked, kein Netz-Zugriff."""
from unittest.mock import MagicMock, patch

import pytest

from kern.typen import Befund, Zustand
from agenten.agent1_bewertung import (
    SCHWELLEN,
    _fcf_yield,
    _kbv,
    _kgv,
    _roe,
    _schulden,
    analyse,
    conviction,
)

VOLLSTAENDIGE_INFO = {
    "trailingPE": 12.0,
    "priceToBook": 1.5,
    "returnOnEquity": 0.20,
    "freeCashflow": 2_000_000_000,
    "marketCap": 40_000_000_000,
    "totalDebt": 5_000_000_000,
    "totalStockholderEquity": 10_000_000_000,
}


class TestKGV:
    def test_unter_schwelle_besteht(self):
        b = _kgv(SCHWELLEN["kgv_max"] - 1)
        assert b.bestanden is True

    def test_ueber_schwelle_besteht_nicht(self):
        b = _kgv(SCHWELLEN["kgv_max"] + 1)
        assert b.bestanden is False

    def test_negativ_ist_unbestimmt(self):
        b = _kgv(-5.0)
        assert b.bestanden is None
        assert b.zustand == Zustand.UNBESTIMMT

    def test_none_ist_unbestimmt(self):
        b = _kgv(None)
        assert b.bestanden is None

    def test_gibt_befund_zurueck_nicht_zahl(self):
        b = _kgv(15.0)
        assert isinstance(b, Befund)


class TestKBV:
    def test_unter_schwelle_besteht(self):
        assert _kbv(SCHWELLEN["kbv_max"] - 0.5).bestanden is True

    def test_ueber_schwelle_besteht_nicht(self):
        assert _kbv(SCHWELLEN["kbv_max"] + 0.5).bestanden is False

    def test_none_ist_unbestimmt(self):
        assert _kbv(None).bestanden is None


class TestROE:
    def test_ueber_schwelle_besteht(self):
        assert _roe(SCHWELLEN["roe_min"] + 0.05).bestanden is True

    def test_unter_schwelle_besteht_nicht(self):
        assert _roe(SCHWELLEN["roe_min"] - 0.05).bestanden is False

    def test_none_ist_unbestimmt(self):
        assert _roe(None).bestanden is None


class TestFCFYield:
    def test_hoch_genug_besteht(self):
        fcf = 1_000_000
        cap = 10_000_000  # Yield = 10%
        assert _fcf_yield(fcf, cap).bestanden is True

    def test_zu_niedrig_besteht_nicht(self):
        fcf = 100_000
        cap = 100_000_000  # Yield = 0.1%
        assert _fcf_yield(fcf, cap).bestanden is False

    def test_fehlender_fcf_ist_unbestimmt(self):
        assert _fcf_yield(None, 1_000_000).bestanden is None

    def test_fehlende_marktcap_ist_unbestimmt(self):
        assert _fcf_yield(1_000_000, None).bestanden is None

    def test_marktcap_null_ist_unbestimmt(self):
        assert _fcf_yield(1_000_000, 0).bestanden is None


class TestSchulden:
    def test_niedrig_besteht(self):
        assert _schulden(5_000_000, 10_000_000).bestanden is True  # ratio 0.5

    def test_hoch_besteht_nicht(self):
        assert _schulden(20_000_000, 10_000_000).bestanden is False  # ratio 2.0

    def test_equity_null_ist_unbestimmt(self):
        assert _schulden(1_000_000, 0).bestanden is None


class TestAnalyse:
    def _mock_ticker(self, info: dict):
        mock = MagicMock()
        mock.info = info
        return mock

    def test_gibt_nur_befund_objekte_zurueck(self):
        with patch("agenten.agent1_bewertung.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker(VOLLSTAENDIGE_INFO)
            befunde = analyse("TEST")
        for b in befunde:
            assert isinstance(b, Befund), "CLAUDE.md: kein Agent gibt nackte Zahl zurück"

    def test_leere_info_alles_unbestimmt(self):
        with patch("agenten.agent1_bewertung.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker({})
            befunde = analyse("TEST")
        for b in befunde:
            assert b.bestanden is None, "Fehlende Daten → UNBESTIMMT"
            assert b.zustand != Zustand.PASSIERT, "Fehlende Daten → nie PASSIERT"

    def test_exception_gibt_unbestimmt(self):
        with patch("agenten.agent1_bewertung.yf.Ticker") as mock_cls:
            mock_cls.side_effect = RuntimeError("Netz")
            befunde = analyse("TEST")
        assert len(befunde) >= 1
        assert all(b.bestanden is None for b in befunde)

    def test_vollstaendige_daten_liefern_bewertungen(self):
        with patch("agenten.agent1_bewertung.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker(VOLLSTAENDIGE_INFO)
            befunde = analyse("TEST")
        bewertbar = [b for b in befunde if b.bestanden is not None]
        assert len(bewertbar) > 0


class TestConviction:
    def test_alle_bestanden_gibt_100(self):
        befunde = [Befund(label=f"X{i}", bestanden=True) for i in range(5)]
        assert conviction(befunde) == 100.0

    def test_keine_bestanden_gibt_0(self):
        befunde = [Befund(label=f"X{i}", bestanden=False) for i in range(5)]
        assert conviction(befunde) == 0.0

    def test_alle_unbestimmt_gibt_0(self):
        befunde = [Befund(label=f"X{i}", bestanden=None) for i in range(5)]
        assert conviction(befunde) == 0.0

    def test_haelfte_bestanden_gibt_50(self):
        befunde = [
            Befund(label="A", bestanden=True),
            Befund(label="B", bestanden=False),
        ]
        assert conviction(befunde) == 50.0

    def test_unbestimmt_zaehlt_nicht_mit(self):
        befunde = [
            Befund(label="A", bestanden=True),
            Befund(label="B", bestanden=None),  # nicht mitzählen
        ]
        assert conviction(befunde) == 100.0

    def test_schwellen_kommen_aus_dict(self):
        assert "kgv_max" in SCHWELLEN, "Schwellen müssen im SCHWELLEN-Dict stehen"
        assert "conviction_gewicht" in SCHWELLEN

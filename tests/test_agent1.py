"""Tests für Agent 1 – Bewertung. Mocked, kein Netz-Zugriff."""
from unittest.mock import MagicMock, patch

import pandas as pd
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


def _make_balance_sheet(stockholders_equity: float) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=1)
    return pd.DataFrame({"Stockholders Equity": [stockholders_equity]}, index=dates).T


def _make_cashflow(fcf: float) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=3)
    return pd.DataFrame({"Free Cash Flow": [fcf, fcf * 0.9, fcf * 0.8]}, index=dates).T


class TestAnalyse:
    def _mock_ticker(self, info: dict, balance_sheet=None, cashflow=None):
        mock = MagicMock()
        mock.info = info
        mock.balance_sheet = balance_sheet if balance_sheet is not None else pd.DataFrame()
        mock.cashflow = cashflow if cashflow is not None else pd.DataFrame()
        return mock

    def test_gibt_nur_befund_objekte_zurueck(self):
        with patch("agenten.agent1_bewertung.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker(
                VOLLSTAENDIGE_INFO,
                _make_balance_sheet(10e9),
                _make_cashflow(2e9),
            )
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
            mock_cls.return_value = self._mock_ticker(
                VOLLSTAENDIGE_INFO,
                _make_balance_sheet(10e9),
                _make_cashflow(2e9),
            )
            befunde = analyse("TEST")
        bewertbar = [b for b in befunde if b.bestanden is not None]
        assert len(bewertbar) > 0

    # --- Bug-Regressionstests (Bugs zuerst als Test, dann behoben) ---

    def test_schulden_nicht_unbestimmt_wenn_equity_im_balance_sheet(self):
        """Bug: totalStockholderEquity=None in info, aber Wert im balance_sheet vorhanden."""
        info_ohne_equity = {
            "trailingPE": 15.0,
            "priceToBook": 1.5,
            "returnOnEquity": 0.15,
            "marketCap": 1_000_000_000_000,
            "totalDebt": 50_000_000_000,
            "totalStockholderEquity": None,  # ← Bug: info gibt None zurück
        }
        with patch("agenten.agent1_bewertung.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker(
                info_ohne_equity,
                _make_balance_sheet(200_000_000_000),  # ← steht aber im balance_sheet
                _make_cashflow(5e9),
            )
            befunde = analyse("TEST")
        schulden_b = next(b for b in befunde if "Schulden" in b.label)
        assert schulden_b.bestanden is not None, (
            "Schulden/EK muss bewertbar sein wenn Daten im balance_sheet vorliegen"
        )

    def test_fcf_rendite_aus_cashflow_statement_nicht_aus_info(self):
        """Bug: info['freeCashflow'] liefert falschen Wert — cashflow Statement ist korrekt."""
        info_falscher_fcf = {
            "marketCap": 3_000_000_000_000,  # 3 Bio
            "freeCashflow": 1_000_000_000,    # ← falsch: 1 Mrd statt 70 Mrd
        }
        korrekter_fcf = 70_000_000_000  # ← cashflow Statement: 70 Mrd
        with patch("agenten.agent1_bewertung.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker(
                info_falscher_fcf,
                pd.DataFrame(),
                _make_cashflow(korrekter_fcf),
            )
            befunde = analyse("TEST")
        fcf_b = next(b for b in befunde if "FCF" in b.label)
        # Yield aus cashflow: 70B / 3000B ≈ 2.33% ≥ Schwelle 3%? Knapp drunter → False
        # Yield aus info (falsch): 1B / 3000B = 0.033% → auch False, aber aus falschen Daten
        # Was wir prüfen: wert muss ~2.33% sein, nicht ~0.033%
        assert fcf_b.wert is not None
        assert fcf_b.wert > 0.02, (
            f"FCF-Rendite muss aus cashflow Statement kommen (erwartet ~2.3%, bekommen {fcf_b.wert:.4f})"
        )


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

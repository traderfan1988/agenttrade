"""Tests für Agent 3 – Qualitätsanalyse. Mocked, kein Netz-Zugriff."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from kern.typen import Befund, Zustand
from agenten.agent3_qualitaet import (
    SCHWELLEN,
    _fcf_konsistenz,
    _gewinnqualitaet,
    _margen_stabilitaet,
    _roa,
    analyse,
    conviction,
)


def _cashflow(fcf_werte: list) -> pd.DataFrame:
    """Mock cashflow DataFrame im yfinance-Format (Index=Items, Columns=Dates)."""
    dates = pd.date_range("2021-01-01", periods=len(fcf_werte), freq="YE")
    return pd.DataFrame({"Free Cash Flow": fcf_werte}, index=dates).T


def _financials(gross: list, revenue: list, net: list) -> pd.DataFrame:
    """Mock financials DataFrame im yfinance-Format."""
    n = min(len(gross), len(revenue), len(net))
    dates = pd.date_range("2021-01-01", periods=n, freq="YE")
    return pd.DataFrame(
        {"Gross Profit": gross[:n], "Total Revenue": revenue[:n], "Net Income": net[:n]},
        index=dates,
    ).T


class TestFCFKonsistenz:
    def test_alle_positiv_besteht(self):
        cf = _cashflow([100, 200, 300, 400])
        b = _fcf_konsistenz(cf)
        assert b.bestanden is True

    def test_ein_negatives_besteht_nicht(self):
        cf = _cashflow([100, -50, 300, 400])
        b = _fcf_konsistenz(cf)
        assert b.bestanden is False

    def test_zu_wenig_jahre_ist_unbestimmt(self):
        cf = _cashflow([100, 200])  # nur 2 Jahre, Schwelle = 3
        b = _fcf_konsistenz(cf)
        assert b.bestanden is None

    def test_leerer_df_ist_unbestimmt(self):
        b = _fcf_konsistenz(pd.DataFrame())
        assert b.bestanden is None
        assert b.zustand == Zustand.UNBESTIMMT

    def test_fehlende_zeile_ist_unbestimmt(self):
        cf = pd.DataFrame({"Kein FCF": [1, 2, 3]}).T
        b = _fcf_konsistenz(cf)
        assert b.bestanden is None

    def test_gibt_befund_zurueck(self):
        cf = _cashflow([100, 200, 300])
        assert isinstance(_fcf_konsistenz(cf), Befund)


class TestGewinnqualitaet:
    def test_hohe_qualitaet_besteht(self):
        cf = _cashflow([90, 80, 70, 60])
        fin = _financials([300, 280, 260, 240], [400, 380, 360, 340], [100, 90, 80, 70])
        b = _gewinnqualitaet(cf, fin)
        assert b.bestanden is True  # FCF/NetIncome ≈ 90%

    def test_niedrige_qualitaet_besteht_nicht(self):
        cf = _cashflow([10, 8, 7, 6])
        fin = _financials([300, 280, 260, 240], [400, 380, 360, 340], [100, 90, 80, 70])
        b = _gewinnqualitaet(cf, fin)
        assert b.bestanden is False  # FCF/NetIncome ≈ 9%

    def test_negativer_nettogewinn_wird_ignoriert(self):
        cf = _cashflow([90, 80, -50, 60])
        # Jahr mit -50 Net Income wird ignoriert
        fin = _financials(
            [300, 280, 260, 240], [400, 380, 360, 340], [100, 90, -50, 70]
        )
        b = _gewinnqualitaet(cf, fin)
        # Nur Jahre mit positivem NetIncome werden gewertet
        assert b.bestanden is not None  # muss bewertbar sein

    def test_leere_daten_sind_unbestimmt(self):
        b = _gewinnqualitaet(pd.DataFrame(), pd.DataFrame())
        assert b.bestanden is None


class TestMargenStabilitaet:
    def test_stabile_marge_besteht(self):
        fin = _financials(
            [300, 300, 300, 300],  # Gross Profit konstant
            [1000, 1000, 1000, 1000],  # Revenue konstant
            [100, 100, 100, 100],
        )
        b = _margen_stabilitaet(fin)
        assert b.bestanden is True  # std = 0

    def test_volatile_marge_besteht_nicht(self):
        fin = _financials(
            [100, 500, 100, 500],  # stark schwankend
            [1000, 1000, 1000, 1000],
            [50, 50, 50, 50],
        )
        b = _margen_stabilitaet(fin)
        assert b.bestanden is False

    def test_zu_wenig_daten_ist_unbestimmt(self):
        fin = _financials([300], [1000], [100])  # nur 1 Jahr
        b = _margen_stabilitaet(fin)
        assert b.bestanden is None

    def test_wert_ist_durchschnittsmarge(self):
        fin = _financials([300, 300], [1000, 1000], [100, 100])
        b = _margen_stabilitaet(fin)
        assert abs(b.wert - 0.30) < 1e-6  # 30% Marge


class TestROA:
    def test_hoch_genug_besteht(self):
        b = _roa(SCHWELLEN["roa_min"] + 0.02)
        assert b.bestanden is True

    def test_zu_niedrig_besteht_nicht(self):
        b = _roa(SCHWELLEN["roa_min"] - 0.01)
        assert b.bestanden is False

    def test_none_ist_unbestimmt(self):
        b = _roa(None)
        assert b.bestanden is None
        assert b.zustand == Zustand.UNBESTIMMT

    def test_gibt_befund_zurueck(self):
        assert isinstance(_roa(0.10), Befund)


class TestAnalyse:
    def _mock_ticker(self, info: dict, cashflow: pd.DataFrame, financials: pd.DataFrame):
        mock = MagicMock()
        mock.info = info
        mock.cashflow = cashflow
        mock.financials = financials
        return mock

    def test_gibt_nur_befund_objekte_zurueck(self):
        cf = _cashflow([90, 80, 70, 60])
        fin = _financials([300, 280, 260, 240], [1000, 980, 960, 940], [100, 90, 80, 70])
        info = {"returnOnAssets": 0.12}
        with patch("agenten.agent3_qualitaet.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker(info, cf, fin)
            befunde = analyse("TEST")
        for b in befunde:
            assert isinstance(b, Befund), "CLAUDE.md: kein Agent gibt nackte Zahl zurück"

    def test_leere_cashflow_gibt_unbestimmt(self):
        with patch("agenten.agent3_qualitaet.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker(
                {}, pd.DataFrame(), pd.DataFrame()
            )
            befunde = analyse("TEST")
        fcf_befund = befunde[0]  # FCF-Konsistenz
        assert fcf_befund.bestanden is None

    def test_exception_gibt_unbestimmt(self):
        with patch("agenten.agent3_qualitaet.yf.Ticker") as mock_cls:
            mock_cls.side_effect = RuntimeError("Netz")
            befunde = analyse("TEST")
        assert all(b.bestanden is None for b in befunde)

    def test_fehlende_daten_nie_passiert(self):
        with patch("agenten.agent3_qualitaet.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker({}, pd.DataFrame(), pd.DataFrame())
            befunde = analyse("TEST")
        for b in befunde:
            if b.bestanden is None:
                assert b.zustand != Zustand.PASSIERT, "Fehlende Daten → nie PASSIERT"


class TestConviction:
    def test_alle_bestanden_gibt_100(self):
        befunde = [Befund(label=f"X{i}", bestanden=True) for i in range(4)]
        assert conviction(befunde) == 100.0

    def test_keine_bewertbaren_gibt_0(self):
        befunde = [Befund(label="X", bestanden=None)]
        assert conviction(befunde) == 0.0

    def test_schwellen_kommen_aus_dict(self):
        assert "fcf_positiv_jahre_min" in SCHWELLEN
        assert "gewinnqualitaet_min" in SCHWELLEN
        assert "marge_std_max" in SCHWELLEN
        assert "roa_min" in SCHWELLEN
        assert "conviction_gewicht" in SCHWELLEN

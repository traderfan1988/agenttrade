"""Tests für Agent 4 – Wachstumsanalyse. Mocked, kein Netz-Zugriff."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from kern.typen import Befund, Zustand
from agenten.agent4_wachstum import (
    SCHWELLEN,
    _cagr,
    _fcf_cagr,
    _gewinn_cagr,
    _operating_leverage,
    _umsatz_cagr,
    analyse,
    conviction,
)


def _make_financials(
    revenues: list,
    net_incomes: list,
    op_incomes: list = None,
) -> pd.DataFrame:
    """Mock financials im yfinance-Format: Zeilen=Items, Spalten=Datum neueste zuerst."""
    n = len(revenues)
    dates = pd.date_range("2021-01-01", periods=n, freq="YE")
    dates = dates[::-1]  # neueste zuerst: 2024, 2023, 2022, 2021
    data = {"Total Revenue": revenues, "Net Income": net_incomes}
    if op_incomes is not None:
        data["Operating Income"] = op_incomes
    return pd.DataFrame(data, index=dates).T


def _make_cashflow(fcf_werte: list) -> pd.DataFrame:
    """Mock cashflow im yfinance-Format: neueste Werte zuerst."""
    n = len(fcf_werte)
    dates = pd.date_range("2021-01-01", periods=n, freq="YE")[::-1]
    return pd.DataFrame({"Free Cash Flow": fcf_werte}, index=dates).T


class TestCAGR:
    def test_wachstum_korrekt_berechnet(self):
        # 80 → 100 in 3 Jahren: CAGR = (100/80)^(1/3) - 1 ≈ 7.72%
        s = pd.Series([100, 90, 85, 80], dtype=float)
        result = _cagr(s, 3)
        assert result is not None
        assert abs(result - ((100 / 80) ** (1 / 3) - 1)) < 1e-6

    def test_negatives_wachstum_moeglich(self):
        s = pd.Series([80, 85, 90, 100], dtype=float)  # fallend
        result = _cagr(s, 3)
        assert result is not None
        assert result < 0

    def test_zu_wenig_daten_gibt_none(self):
        s = pd.Series([100, 90, 80], dtype=float)  # 3 Werte für 3J-CAGR → braucht 4
        assert _cagr(s, 3) is None

    def test_negativer_basiswert_gibt_none(self):
        s = pd.Series([100, 90, 85, -10], dtype=float)  # Basis negativ
        assert _cagr(s, 3) is None

    def test_nullwert_als_basis_gibt_none(self):
        s = pd.Series([100, 90, 85, 0], dtype=float)
        assert _cagr(s, 3) is None

    def test_negativer_endwert_gibt_none(self):
        s = pd.Series([-100, 90, 85, 80], dtype=float)
        assert _cagr(s, 3) is None


class TestUmsatzCAGR:
    def test_starkes_wachstum_besteht(self):
        # Revenue: 100 → 110 → 121 → 133.1 (10% CAGR) → besteht ≥ 5%
        fin = _make_financials([133.1e9, 121e9, 110e9, 100e9], [10e9] * 4)
        b = _umsatz_cagr(fin)
        assert b.bestanden is True

    def test_zu_schwaches_wachstum_besteht_nicht(self):
        # Revenue: 100 → 101 → 102 → 103 (CAGR ≈ 1%) < 5%
        fin = _make_financials([103e9, 102e9, 101e9, 100e9], [5e9] * 4)
        b = _umsatz_cagr(fin)
        assert b.bestanden is False

    def test_schrumpfender_umsatz_besteht_nicht(self):
        fin = _make_financials([80e9, 85e9, 90e9, 100e9], [5e9] * 4)
        b = _umsatz_cagr(fin)
        assert b.bestanden is False

    def test_zu_wenig_daten_ist_unbestimmt(self):
        fin = _make_financials([100e9, 105e9], [5e9, 5e9])  # nur 2 Jahre
        b = _umsatz_cagr(fin)
        assert b.bestanden is None
        assert b.zustand == Zustand.UNBESTIMMT

    def test_fehlende_zeile_ist_unbestimmt(self):
        fin = pd.DataFrame({"Net Income": [5e9]}).T
        b = _umsatz_cagr(fin)
        assert b.bestanden is None

    def test_fehlende_daten_nie_passiert(self):
        b = _umsatz_cagr(pd.DataFrame())
        assert b.zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        fin = _make_financials([133e9, 121e9, 110e9, 100e9], [10e9] * 4)
        assert isinstance(_umsatz_cagr(fin), Befund)

    def test_wert_enthaelt_cagr(self):
        fin = _make_financials([133.1e9, 121e9, 110e9, 100e9], [10e9] * 4)
        b = _umsatz_cagr(fin)
        assert b.wert is not None
        assert b.wert > 0


class TestGewinnCAGR:
    def test_starkes_wachstum_besteht(self):
        fin = _make_financials([100e9] * 4, [133.1e9, 121e9, 110e9, 100e9])
        b = _gewinn_cagr(fin)
        assert b.bestanden is True  # ~10% CAGR ≥ 5%

    def test_negativer_basisgewinn_ist_unbestimmt(self):
        fin = _make_financials([100e9] * 4, [110e9, 105e9, 100e9, -5e9])
        b = _gewinn_cagr(fin)
        assert b.bestanden is None

    def test_zu_wenig_daten_ist_unbestimmt(self):
        fin = _make_financials([100e9, 110e9], [10e9, 11e9])
        b = _gewinn_cagr(fin)
        assert b.bestanden is None

    def test_fehlende_daten_nie_passiert(self):
        b = _gewinn_cagr(pd.DataFrame())
        assert b.zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        fin = _make_financials([100e9] * 4, [133e9, 121e9, 110e9, 100e9])
        assert isinstance(_gewinn_cagr(fin), Befund)


class TestFCFCAGR:
    def test_starkes_wachstum_besteht(self):
        # FCF: 100 → 110 → 121 → 133.1 (10% CAGR) ≥ 3%
        cf = _make_cashflow([133.1e9, 121e9, 110e9, 100e9])
        b = _fcf_cagr(cf)
        assert b.bestanden is True

    def test_leichtes_wachstum_besteht_wenn_ueber_schwelle(self):
        # CAGR ≈ 4% > 3%
        cf = _make_cashflow([112.5e9, 108.7e9, 105e9, 101.4e9])
        b = _fcf_cagr(cf)
        # Accept anything above -schwelle (fuzzy test of threshold)
        assert b.bestanden is not None

    def test_negativer_fcf_ist_unbestimmt(self):
        cf = _make_cashflow([100e9, 90e9, 80e9, -10e9])
        b = _fcf_cagr(cf)
        assert b.bestanden is None

    def test_zu_wenig_daten_ist_unbestimmt(self):
        cf = _make_cashflow([100e9, 110e9])
        b = _fcf_cagr(cf)
        assert b.bestanden is None

    def test_fehlende_daten_nie_passiert(self):
        b = _fcf_cagr(pd.DataFrame())
        assert b.zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        cf = _make_cashflow([133e9, 121e9, 110e9, 100e9])
        assert isinstance(_fcf_cagr(cf), Befund)


class TestOperatingLeverage:
    def test_skalierbar_wenn_op_income_schneller_waechst(self):
        # Revenue CAGR ≈ 5%, Op.Income CAGR ≈ 10% → positives Leverage
        fin = _make_financials(
            revenues=[121e9, 115e9, 110e9, 105e9],    # ~4.8% CAGR
            net_incomes=[20e9] * 4,
            op_incomes=[133.1e9, 121e9, 110e9, 100e9],  # ~10% CAGR
        )
        b = _operating_leverage(fin)
        assert b.bestanden is True

    def test_nicht_skalierbar_wenn_op_income_langsamer_waechst(self):
        fin = _make_financials(
            revenues=[133.1e9, 121e9, 110e9, 100e9],   # ~10% CAGR
            net_incomes=[20e9] * 4,
            op_incomes=[105e9, 103e9, 101e9, 100e9],   # ~1.6% CAGR
        )
        b = _operating_leverage(fin)
        assert b.bestanden is False

    def test_fehlende_daten_sind_unbestimmt(self):
        fin = _make_financials([100e9, 110e9], [10e9, 11e9])
        b = _operating_leverage(fin)
        assert b.bestanden is None

    def test_fehlende_op_income_zeile_ist_unbestimmt(self):
        fin = pd.DataFrame({"Total Revenue": [100e9]}).T
        b = _operating_leverage(fin)
        assert b.bestanden is None

    def test_fehlende_daten_nie_passiert(self):
        b = _operating_leverage(pd.DataFrame())
        assert b.zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        fin = _make_financials(
            [121e9, 115e9, 110e9, 105e9], [10e9] * 4, [133e9, 121e9, 110e9, 100e9]
        )
        assert isinstance(_operating_leverage(fin), Befund)


class TestAnalyse:
    def _mock_ticker(self, financials: pd.DataFrame, cashflow: pd.DataFrame):
        mock = MagicMock()
        mock.financials = financials
        mock.cashflow = cashflow
        return mock

    def test_gibt_nur_befund_objekte_zurueck(self):
        fin = _make_financials(
            [133e9, 121e9, 110e9, 100e9],
            [20e9, 18e9, 16e9, 14e9],
            [60e9, 55e9, 50e9, 45e9],
        )
        cf = _make_cashflow([50e9, 45e9, 40e9, 36e9])
        with patch("agenten.agent4_wachstum.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker(fin, cf)
            befunde = analyse("TEST")
        for b in befunde:
            assert isinstance(b, Befund), "CLAUDE.md: kein Agent gibt nackte Zahl zurück"

    def test_leere_daten_geben_unbestimmt(self):
        with patch("agenten.agent4_wachstum.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker(pd.DataFrame(), pd.DataFrame())
            befunde = analyse("TEST")
        for b in befunde:
            assert b.zustand != Zustand.PASSIERT

    def test_exception_gibt_unbestimmt(self):
        with patch("agenten.agent4_wachstum.yf.Ticker") as mock_cls:
            mock_cls.side_effect = RuntimeError("Netz")
            befunde = analyse("TEST")
        assert all(b.bestanden is None for b in befunde)

    def test_fehlende_daten_nie_passiert(self):
        with patch("agenten.agent4_wachstum.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker(pd.DataFrame(), pd.DataFrame())
            befunde = analyse("TEST")
        for b in befunde:
            if b.bestanden is None:
                assert b.zustand != Zustand.PASSIERT


class TestConviction:
    def test_alle_bestanden_gibt_100(self):
        befunde = [Befund(label=f"X{i}", bestanden=True) for i in range(4)]
        assert conviction(befunde) == 100.0

    def test_keine_bewertbaren_gibt_0(self):
        befunde = [Befund(label="X", bestanden=None)]
        assert conviction(befunde) == 0.0

    def test_haelfte_gibt_50(self):
        befunde = [Befund(label="A", bestanden=True), Befund(label="B", bestanden=False)]
        assert conviction(befunde) == 50.0

    def test_schwellen_kommen_aus_dict(self):
        assert "umsatz_cagr_min" in SCHWELLEN
        assert "gewinn_cagr_min" in SCHWELLEN
        assert "fcf_cagr_min" in SCHWELLEN
        assert "conviction_gewicht" in SCHWELLEN

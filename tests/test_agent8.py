"""Tests für Agent 8 – Bilanzqualität. Mocked, kein Netz-Zugriff."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from kern.typen import Befund, Zustand
from agenten.agent8_bilanzqualitaet import (
    SCHWELLEN,
    _current_ratio,
    _ebitda_aus_financials,
    _goodwill_anteil,
    _net_debt_ebitda,
    _working_capital,
    analyse,
    conviction,
)


def _make_balance_sheet(**kwargs) -> pd.DataFrame:
    if not kwargs:
        return pd.DataFrame()
    return pd.DataFrame({k: [v] for k, v in kwargs.items()}).T


def _make_financials(**kwargs) -> pd.DataFrame:
    if not kwargs:
        return pd.DataFrame()
    return pd.DataFrame({k: [v] for k, v in kwargs.items()}).T


class TestNetDebtEbitda:
    def test_niedriger_hebel_besteht(self):
        # Net Debt = 100K, EBITDA = 200K → 0.5× ≤ 3× → PASSIERT
        b = _net_debt_ebitda(total_debt=500_000, cash=400_000, ebitda=200_000)
        assert b.bestanden is True

    def test_hoher_hebel_besteht_nicht(self):
        # Net Debt = 900K, EBITDA = 200K → 4.5× > 3×
        b = _net_debt_ebitda(total_debt=1_000_000, cash=100_000, ebitda=200_000)
        assert b.bestanden is False

    def test_nettoliquid_ist_passiert(self):
        # Cash > Debt → Nettoliquid
        b = _net_debt_ebitda(total_debt=100_000, cash=500_000, ebitda=200_000)
        assert b.bestanden is True

    def test_negativer_ebitda_ist_unbestimmt(self):
        b = _net_debt_ebitda(total_debt=500_000, cash=100_000, ebitda=-100_000)
        assert b.bestanden is None

    def test_kein_ebitda_ist_unbestimmt(self):
        b = _net_debt_ebitda(total_debt=500_000, cash=100_000, ebitda=None)
        assert b.bestanden is None

    def test_kein_debt_kein_cash_ist_unbestimmt(self):
        b = _net_debt_ebitda(total_debt=None, cash=None, ebitda=200_000)
        assert b.bestanden is None

    def test_fehlende_daten_nie_passiert(self):
        assert _net_debt_ebitda(None, None, None).zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        assert isinstance(_net_debt_ebitda(500_000, 100_000, 300_000), Befund)


class TestCurrentRatio:
    def test_hohe_liquiditaet_besteht(self):
        b = _current_ratio(current_assets=3_000_000, current_liabilities=1_000_000)
        # ratio = 3.0 ≥ 1.5 → PASSIERT
        assert b.bestanden is True

    def test_niedrige_liquiditaet_besteht_nicht(self):
        b = _current_ratio(current_assets=1_000_000, current_liabilities=2_000_000)
        # ratio = 0.5 < 1.5 → NICHT_PASSIERT
        assert b.bestanden is False

    def test_genau_schwelle_besteht(self):
        schwelle = SCHWELLEN["current_ratio_min"]
        b = _current_ratio(
            current_assets=int(schwelle * 1_000_000),
            current_liabilities=1_000_000,
        )
        assert b.bestanden is True

    def test_none_assets_ist_unbestimmt(self):
        b = _current_ratio(current_assets=None, current_liabilities=1_000_000)
        assert b.bestanden is None

    def test_fehlende_daten_nie_passiert(self):
        assert _current_ratio(None, None).zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        assert isinstance(_current_ratio(3_000_000, 1_000_000), Befund)


class TestGoodwillAnteil:
    def test_niedriger_anteil_besteht(self):
        # Goodwill = 10%, Total Assets = 100%
        b = _goodwill_anteil(goodwill=100_000, total_assets=1_000_000)
        assert b.bestanden is True

    def test_hoher_anteil_besteht_nicht(self):
        # Goodwill = 50%
        b = _goodwill_anteil(goodwill=500_000, total_assets=1_000_000)
        assert b.bestanden is False

    def test_kein_goodwill_besteht(self):
        b = _goodwill_anteil(goodwill=0, total_assets=1_000_000)
        assert b.bestanden is True

    def test_none_goodwill_ist_unbestimmt(self):
        # Fehlende Goodwill-Daten → nicht bewertbar (z.B. bei Finanzunternehmen)
        b = _goodwill_anteil(goodwill=None, total_assets=1_000_000)
        assert b.bestanden is None

    def test_none_total_assets_ist_unbestimmt(self):
        b = _goodwill_anteil(goodwill=100_000, total_assets=None)
        assert b.bestanden is None

    def test_fehlende_daten_nie_passiert(self):
        assert _goodwill_anteil(None, None).zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        assert isinstance(_goodwill_anteil(100_000, 1_000_000), Befund)


class TestWorkingCapital:
    def test_positives_working_capital_besteht(self):
        b = _working_capital(current_assets=2_000_000, current_liabilities=500_000)
        assert b.bestanden is True

    def test_negatives_working_capital_besteht_nicht(self):
        b = _working_capital(current_assets=500_000, current_liabilities=2_000_000)
        assert b.bestanden is False

    def test_null_working_capital_besteht_nicht(self):
        b = _working_capital(current_assets=1_000_000, current_liabilities=1_000_000)
        assert b.bestanden is False

    def test_none_ist_unbestimmt(self):
        b = _working_capital(current_assets=None, current_liabilities=1_000_000)
        assert b.bestanden is None

    def test_fehlende_daten_nie_passiert(self):
        assert _working_capital(None, None).zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        assert isinstance(_working_capital(2_000_000, 1_000_000), Befund)


class TestEbitdaHelfer:
    def test_liest_ebitda_feld(self):
        fin = _make_financials(EBITDA=500_000)
        assert _ebitda_aus_financials(fin) == pytest.approx(500_000)

    def test_leerer_df_gibt_none(self):
        assert _ebitda_aus_financials(pd.DataFrame()) is None

    def test_none_gibt_none(self):
        assert _ebitda_aus_financials(None) is None


class TestAnalyse:
    def _mock_ticker(self, info, balance_sheet, financials):
        mock = MagicMock()
        mock.info = info
        mock.balance_sheet = balance_sheet
        mock.financials = financials
        return mock

    def test_gibt_nur_befund_objekte_zurueck(self):
        bs = _make_balance_sheet(
            **{
                "Total Current Assets": 3_000_000,
                "Total Current Liabilities": 1_000_000,
                "Goodwill": 100_000,
                "Total Assets": 5_000_000,
            }
        )
        fin = _make_financials(EBITDA=500_000)
        info = {"totalDebt": 300_000, "totalCash": 100_000}
        with patch("agenten.agent8_bilanzqualitaet.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker(info, bs, fin)
            befunde = analyse("TEST")
        for b in befunde:
            assert isinstance(b, Befund)

    def test_leere_daten_geben_unbestimmt(self):
        with patch("agenten.agent8_bilanzqualitaet.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker({}, pd.DataFrame(), pd.DataFrame())
            befunde = analyse("TEST")
        for b in befunde:
            assert b.zustand != Zustand.PASSIERT

    def test_exception_gibt_unbestimmt(self):
        with patch("agenten.agent8_bilanzqualitaet.yf.Ticker") as mock_cls:
            mock_cls.side_effect = RuntimeError("Netz")
            befunde = analyse("TEST")
        assert all(b.bestanden is None for b in befunde)

    def test_fehlende_daten_nie_passiert(self):
        with patch("agenten.agent8_bilanzqualitaet.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker({}, pd.DataFrame(), pd.DataFrame())
            befunde = analyse("TEST")
        for b in befunde:
            if b.bestanden is None:
                assert b.zustand != Zustand.PASSIERT


class TestConviction:
    def test_alle_bestanden_gibt_100(self):
        befunde = [Befund(label=f"X{i}", bestanden=True) for i in range(4)]
        assert conviction(befunde) == 100.0

    def test_keine_bewertbaren_gibt_0(self):
        assert conviction([Befund(label="X", bestanden=None)]) == 0.0

    def test_haelfte_gibt_50(self):
        befunde = [Befund(label="A", bestanden=True), Befund(label="B", bestanden=False)]
        assert conviction(befunde) == 50.0

    def test_schwellen_enthalten_pflichtfelder(self):
        assert "net_debt_ebitda_max" in SCHWELLEN
        assert "current_ratio_min" in SCHWELLEN
        assert "goodwill_anteil_max" in SCHWELLEN
        assert "conviction_gewicht" in SCHWELLEN

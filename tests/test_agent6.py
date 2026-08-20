"""Tests für Agent 6 – Kapitalrückgabe. Mocked, kein Netz-Zugriff."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from kern.typen import Befund, Zustand
from agenten.agent6_kapitalrueckgabe import (
    SCHWELLEN,
    _buyback_yield,
    _buyback_aus_cashflow,
    _dividenden_aus_cashflow,
    _dividendenrendite,
    _fcf_payout,
    _shareholder_yield,
    analyse,
    conviction,
)


def _make_cashflow(repurchase=None, dividends=None, issuance=None) -> pd.DataFrame:
    rows = {}
    if repurchase is not None:
        rows["Repurchase Of Capital Stock"] = [repurchase]
    if dividends is not None:
        rows["Common Stock Dividends Paid"] = [dividends]
    if issuance is not None:
        rows["Common Stock Issuance"] = [issuance]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).T


class TestDividendenrendite:
    def test_hohe_rendite_besteht(self):
        b = _dividendenrendite(SCHWELLEN["dividenden_rendite_min"] + 0.01)
        assert b.bestanden is True

    def test_niedrige_rendite_besteht_nicht(self):
        b = _dividendenrendite(0.005)
        assert b.bestanden is False

    def test_null_rendite_ist_unbestimmt(self):
        b = _dividendenrendite(0.0)
        assert b.bestanden is None
        assert b.zustand == Zustand.UNBESTIMMT

    def test_none_ist_unbestimmt(self):
        b = _dividendenrendite(None)
        assert b.bestanden is None

    def test_fehlende_daten_nie_passiert(self):
        assert _dividendenrendite(None).zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        assert isinstance(_dividendenrendite(0.03), Befund)

    def test_wert_ist_rendite(self):
        b = _dividendenrendite(0.035)
        assert b.wert == pytest.approx(0.035, rel=1e-4)


class TestFcfPayout:
    def test_niedriger_payout_besteht(self):
        # FCF > Gewinn, Payout niedrig
        b = _fcf_payout(
            fcf=1_000_000, net_income=800_000,
            dividenden_gezahlt=300_000
        )
        assert b.bestanden is True

    def test_hoher_payout_besteht_nicht(self):
        b = _fcf_payout(
            fcf=500_000, net_income=500_000,
            dividenden_gezahlt=450_000
        )
        assert b.bestanden is False

    def test_keine_dividende_ist_unbestimmt(self):
        b = _fcf_payout(fcf=1_000_000, net_income=800_000, dividenden_gezahlt=0)
        assert b.bestanden is None

    def test_none_dividenden_ist_unbestimmt(self):
        b = _fcf_payout(fcf=1_000_000, net_income=800_000, dividenden_gezahlt=None)
        assert b.bestanden is None

    def test_kein_fcf_ist_unbestimmt(self):
        b = _fcf_payout(fcf=None, net_income=800_000, dividenden_gezahlt=200_000)
        assert b.bestanden is None

    def test_fcf_null_ist_unbestimmt(self):
        b = _fcf_payout(fcf=0, net_income=800_000, dividenden_gezahlt=200_000)
        assert b.bestanden is None

    def test_fehlende_daten_nie_passiert(self):
        assert _fcf_payout(None, None, None).zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        assert isinstance(_fcf_payout(1_000_000, 800_000, 300_000), Befund)


class TestBuybackYield:
    def test_hohe_rendite_besteht(self):
        b = _buyback_yield(
            buyback=2_000_000,
            market_cap=100_000_000,
        )
        # 2% ≥ 1.5% → PASSIERT
        assert b.bestanden is True

    def test_niedrige_rendite_besteht_nicht(self):
        b = _buyback_yield(buyback=500_000, market_cap=100_000_000)
        # 0.5% < 1.5% → NICHT_PASSIERT
        assert b.bestanden is False

    def test_null_buyback_besteht_nicht(self):
        b = _buyback_yield(buyback=0, market_cap=100_000_000)
        assert b.bestanden is False

    def test_none_buyback_ist_unbestimmt(self):
        b = _buyback_yield(buyback=None, market_cap=100_000_000)
        assert b.bestanden is None

    def test_none_market_cap_ist_unbestimmt(self):
        b = _buyback_yield(buyback=1_000_000, market_cap=None)
        assert b.bestanden is None

    def test_fehlende_daten_nie_passiert(self):
        assert _buyback_yield(None, None).zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        assert isinstance(_buyback_yield(2_000_000, 100_000_000), Befund)


class TestShareholderYield:
    def test_hohe_gesamt_rendite_besteht(self):
        b = _shareholder_yield(
            dividenden=1_500_000, buyback=2_000_000, market_cap=100_000_000
        )
        # 3.5% ≥ 3% → PASSIERT
        assert b.bestanden is True

    def test_niedrige_gesamt_rendite_besteht_nicht(self):
        b = _shareholder_yield(
            dividenden=500_000, buyback=500_000, market_cap=100_000_000
        )
        # 1% < 3% → NICHT_PASSIERT
        assert b.bestanden is False

    def test_beide_null_besteht_nicht(self):
        b = _shareholder_yield(dividenden=0, buyback=0, market_cap=100_000_000)
        assert b.bestanden is False

    def test_none_market_cap_ist_unbestimmt(self):
        b = _shareholder_yield(dividenden=1_000_000, buyback=1_000_000, market_cap=None)
        assert b.bestanden is None

    def test_fehlende_daten_nie_passiert(self):
        assert _shareholder_yield(None, None, None).zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        assert isinstance(_shareholder_yield(1_000_000, 2_000_000, 100_000_000), Befund)


class TestCashflowHelfer:
    def test_buyback_aus_cashflow_liest_repurchase(self):
        cf = _make_cashflow(repurchase=-2_000_000)
        result = _buyback_aus_cashflow(cf)
        assert result == pytest.approx(2_000_000)

    def test_buyback_aus_leerem_df_ist_none(self):
        assert _buyback_aus_cashflow(pd.DataFrame()) is None

    def test_buyback_aus_none_ist_none(self):
        assert _buyback_aus_cashflow(None) is None

    def test_dividenden_aus_cashflow_liest_dividenden(self):
        cf = _make_cashflow(dividends=-800_000)
        result = _dividenden_aus_cashflow(cf)
        assert result == pytest.approx(800_000)

    def test_dividenden_aus_leerem_df_ist_none(self):
        assert _dividenden_aus_cashflow(pd.DataFrame()) is None

    def test_dividenden_aus_none_ist_none(self):
        assert _dividenden_aus_cashflow(None) is None


class TestAnalyse:
    def _mock_ticker(self, info, cashflow, financials):
        mock = MagicMock()
        mock.info = info
        mock.cashflow = cashflow
        mock.financials = financials
        return mock

    def _make_financials(self, net_income=1_000_000):
        return pd.DataFrame({"Net Income": [net_income]}).T

    def test_gibt_nur_befund_objekte_zurueck(self):
        cf = _make_cashflow(repurchase=-2_000_000, dividends=-500_000)
        fin = self._make_financials()
        info = {"dividendYield": 0.03, "marketCap": 100_000_000, "freeCashflow": 1_500_000}
        with patch("agenten.agent6_kapitalrueckgabe.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker(info, cf, fin)
            befunde = analyse("TEST")
        for b in befunde:
            assert isinstance(b, Befund)

    def test_leere_daten_geben_unbestimmt(self):
        with patch("agenten.agent6_kapitalrueckgabe.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker({}, pd.DataFrame(), pd.DataFrame())
            befunde = analyse("TEST")
        for b in befunde:
            assert b.zustand != Zustand.PASSIERT

    def test_exception_gibt_unbestimmt(self):
        with patch("agenten.agent6_kapitalrueckgabe.yf.Ticker") as mock_cls:
            mock_cls.side_effect = RuntimeError("Netz")
            befunde = analyse("TEST")
        assert all(b.bestanden is None for b in befunde)


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
        assert "dividenden_rendite_min" in SCHWELLEN
        assert "buyback_yield_min" in SCHWELLEN
        assert "shareholder_yield_min" in SCHWELLEN
        assert "conviction_gewicht" in SCHWELLEN

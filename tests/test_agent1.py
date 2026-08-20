"""Tests für Agent 1 – Fundamentale Bewertungs- & Qualitätsanalyse. Mocked, kein Netz-Zugriff."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from kern.typen import Befund, Zustand
from agenten.agent1_bewertung import (
    SCHWELLEN,
    _ccr,
    _ebit_aus_financials,
    _equity_aus_balance_sheet,
    _ev_ebit,
    _fcf_aus_cashflow,
    _fcf_ev_yield,
    _kbv,
    _roce,
    _schulden,
    _zinsdeckung,
    analyse,
    conviction,
)


def _make_balance_sheet(
    equity: float = 10e9,
    assets: float = 25e9,
    current_liab: float = 5e9,
    cash: float = 3e9,
) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=1)
    return pd.DataFrame(
        {
            "Stockholders Equity": [equity],
            "Total Assets": [assets],
            "Current Liabilities": [current_liab],
            "Cash And Cash Equivalents": [cash],
        },
        index=dates,
    ).T


def _make_financials(
    ebit: float = 5e9,
    net_income: float = 4e9,
    interest_expense: float = -500e6,
) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=1)
    return pd.DataFrame(
        {
            "EBIT": [ebit],
            "Net Income": [net_income],
            "Interest Expense": [interest_expense],
        },
        index=dates,
    ).T


def _make_cashflow(fcf: float = 4e9) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=3)
    return pd.DataFrame(
        {"Free Cash Flow": [fcf, fcf * 0.9, fcf * 0.8]},
        index=dates,
    ).T


class TestEVEBIT:
    def test_unter_schwelle_besteht(self):
        # EV = 100B + 0 - 0 = 100B, EBIT = 10B → EV/EBIT = 10 ≤ 15
        b = _ev_ebit(100e9, 0, 0, 10e9)
        assert b.bestanden is True

    def test_ueber_schwelle_besteht_nicht(self):
        # EV = 200B, EBIT = 5B → EV/EBIT = 40 > 15
        b = _ev_ebit(200e9, 0, 0, 5e9)
        assert b.bestanden is False

    def test_negatives_ebit_ist_unbestimmt(self):
        b = _ev_ebit(100e9, 0, 0, -1e9)
        assert b.bestanden is None
        assert b.zustand == Zustand.UNBESTIMMT

    def test_none_ebit_ist_unbestimmt(self):
        b = _ev_ebit(100e9, 0, 0, None)
        assert b.bestanden is None

    def test_none_marketcap_ist_unbestimmt(self):
        b = _ev_ebit(None, 0, 0, 5e9)
        assert b.bestanden is None

    def test_ev_beruecksichtigt_schulden_und_cash(self):
        # EV = 100B + 20B - 10B = 110B, EBIT = 10B → 11.0
        b = _ev_ebit(100e9, 20e9, 10e9, 10e9)
        assert b.wert == pytest.approx(11.0, rel=1e-4)

    def test_keine_schulden_kein_cash(self):
        # EV = MarketCap
        b = _ev_ebit(100e9, None, None, 10e9)
        assert b.wert == pytest.approx(10.0, rel=1e-4)

    def test_gibt_befund_zurueck(self):
        assert isinstance(_ev_ebit(100e9, 0, 0, 10e9), Befund)


class TestFCFEVYield:
    def test_hoch_genug_besteht(self):
        # FCF = 10B, EV = 100B → yield = 10% ≥ 5%
        b = _fcf_ev_yield(10e9, 100e9, 0, 0)
        assert b.bestanden is True

    def test_zu_niedrig_besteht_nicht(self):
        # FCF = 1B, EV = 100B → yield = 1% < 5%
        b = _fcf_ev_yield(1e9, 100e9, 0, 0)
        assert b.bestanden is False

    def test_kein_fcf_ist_unbestimmt(self):
        b = _fcf_ev_yield(None, 100e9, 0, 0)
        assert b.bestanden is None

    def test_kein_marketcap_ist_unbestimmt(self):
        b = _fcf_ev_yield(5e9, None, 0, 0)
        assert b.bestanden is None

    def test_ev_negativ_ist_unbestimmt(self):
        # EV = 100B + 0 - 200B = -100B (viel Cash)
        b = _fcf_ev_yield(5e9, 100e9, 0, 200e9)
        assert b.bestanden is None

    def test_gibt_befund_zurueck(self):
        assert isinstance(_fcf_ev_yield(5e9, 100e9, 0, 0), Befund)


class TestKBV:
    def test_unter_schwelle_besteht(self):
        assert _kbv(SCHWELLEN["kbv_max"] - 0.5).bestanden is True

    def test_ueber_schwelle_besteht_nicht(self):
        assert _kbv(SCHWELLEN["kbv_max"] + 0.5).bestanden is False

    def test_none_ist_unbestimmt(self):
        assert _kbv(None).bestanden is None

    def test_null_ist_unbestimmt(self):
        assert _kbv(0).bestanden is None

    def test_gibt_befund_zurueck(self):
        assert isinstance(_kbv(1.5), Befund)


class TestCCR:
    def test_hoch_genug_besteht(self):
        # FCF = 90M, NetIncome = 100M → CCR = 90% ≥ 80%
        fin = _make_financials(net_income=100e6)
        b = _ccr(90e6, fin)
        assert b.bestanden is True

    def test_zu_niedrig_besteht_nicht(self):
        # FCF = 50M, NetIncome = 100M → CCR = 50% < 80%
        fin = _make_financials(net_income=100e6)
        b = _ccr(50e6, fin)
        assert b.bestanden is False

    def test_kein_fcf_ist_unbestimmt(self):
        b = _ccr(None, _make_financials())
        assert b.bestanden is None

    def test_negativer_nettogewinn_ist_unbestimmt(self):
        fin = _make_financials(net_income=-100e6)
        b = _ccr(80e6, fin)
        assert b.bestanden is None

    def test_leere_financials_sind_unbestimmt(self):
        b = _ccr(80e6, pd.DataFrame())
        assert b.bestanden is None

    def test_fehlende_daten_nie_passiert(self):
        b = _ccr(None, pd.DataFrame())
        assert b.zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        assert isinstance(_ccr(80e6, _make_financials(net_income=100e6)), Befund)


class TestROCE:
    def test_hoch_genug_besteht(self):
        # EBIT = 5B, CE = Assets - CurrLiab = 25B - 5B = 20B → ROCE = 25% ≥ 8%
        bs = _make_balance_sheet(assets=25e9, current_liab=5e9)
        b = _roce(5e9, bs, "Technology")
        assert b.bestanden is True

    def test_zu_niedrig_besteht_nicht(self):
        # EBIT = 1B, CE = 20B → ROCE = 5% < 8%
        bs = _make_balance_sheet(assets=25e9, current_liab=5e9)
        b = _roce(1e9, bs, "Technology")
        assert b.bestanden is False

    def test_finanzsektor_ist_unbestimmt(self):
        bs = _make_balance_sheet()
        b = _roce(5e9, bs, "Financial Services")
        assert b.bestanden is None
        assert b.zustand == Zustand.UNBESTIMMT

    def test_utilities_ist_unbestimmt(self):
        bs = _make_balance_sheet()
        b = _roce(5e9, bs, "Utilities")
        assert b.bestanden is None

    def test_real_estate_ist_unbestimmt(self):
        bs = _make_balance_sheet()
        b = _roce(5e9, bs, "Real Estate")
        assert b.bestanden is None

    def test_kein_ebit_ist_unbestimmt(self):
        b = _roce(None, _make_balance_sheet(), "Technology")
        assert b.bestanden is None

    def test_leere_bilanz_ist_unbestimmt(self):
        b = _roce(5e9, pd.DataFrame(), "Technology")
        assert b.bestanden is None

    def test_fehlende_daten_nie_passiert(self):
        b = _roce(None, pd.DataFrame(), "Technology")
        assert b.zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        assert isinstance(_roce(5e9, _make_balance_sheet(), "Technology"), Befund)


class TestZinsdeckung:
    def test_hoch_genug_besteht(self):
        # EBIT = 10B, Interest = -1B (Aufwand) → Deckung = 10x ≥ 3x
        fin = _make_financials(ebit=10e9, interest_expense=-1e9)
        b = _zinsdeckung(10e9, fin)
        assert b.bestanden is True

    def test_zu_niedrig_besteht_nicht(self):
        # EBIT = 1B, Interest = -2B → Deckung = 0.5x < 3x
        fin = _make_financials(ebit=1e9, interest_expense=-2e9)
        b = _zinsdeckung(1e9, fin)
        assert b.bestanden is False

    def test_nettoliquid_besteht_immer(self):
        # Interest Expense > 0 → Nettoliquid (Zinserträge > Zinsaufwand)
        fin = _make_financials(interest_expense=100e6)  # positiv = Nettoliquid
        b = _zinsdeckung(5e9, fin)
        assert b.bestanden is True
        assert "Nettoliquid" in b.details

    def test_kein_ebit_ist_unbestimmt(self):
        b = _zinsdeckung(None, _make_financials())
        assert b.bestanden is None

    def test_leere_financials_sind_unbestimmt(self):
        b = _zinsdeckung(5e9, pd.DataFrame())
        assert b.bestanden is None

    def test_fehlende_daten_nie_passiert(self):
        b = _zinsdeckung(None, pd.DataFrame())
        assert b.zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        assert isinstance(_zinsdeckung(10e9, _make_financials()), Befund)

    def test_nettoliquid_wert_ist_none(self):
        # Bei Nettoliquidität kein sinnvoller numerischer Wert
        fin = _make_financials(interest_expense=100e6)
        b = _zinsdeckung(5e9, fin)
        assert b.wert is None


class TestSchulden:
    def test_niedrig_besteht(self):
        assert _schulden(5e9, 10e9).bestanden is True  # ratio 0.5

    def test_hoch_besteht_nicht(self):
        assert _schulden(20e9, 10e9).bestanden is False  # ratio 2.0

    def test_equity_null_ist_unbestimmt(self):
        assert _schulden(1e9, 0).bestanden is None

    def test_fehlende_daten_nie_passiert(self):
        assert _schulden(None, None).zustand != Zustand.PASSIERT


class TestHilfsfunktionen:
    def test_ebit_aus_financials_ebit_zeile(self):
        fin = _make_financials(ebit=5e9)
        assert _ebit_aus_financials(fin) == pytest.approx(5e9)

    def test_ebit_fallback_operating_income(self):
        dates = pd.date_range("2024-01-01", periods=1)
        fin = pd.DataFrame({"Operating Income": [3e9]}, index=dates).T
        assert _ebit_aus_financials(fin) == pytest.approx(3e9)

    def test_ebit_leere_df_gibt_none(self):
        assert _ebit_aus_financials(pd.DataFrame()) is None

    def test_fcf_aus_cashflow_gibt_neuesten_wert(self):
        cf = _make_cashflow(10e9)
        assert _fcf_aus_cashflow(cf) == pytest.approx(10e9)

    def test_equity_aus_balance_sheet_stockholders(self):
        bs = _make_balance_sheet(equity=5e9)
        assert _equity_aus_balance_sheet(bs) == pytest.approx(5e9)


class TestAnalyse:
    def _mock_ticker(self, info: dict, balance_sheet=None, cashflow=None, financials=None):
        mock = MagicMock()
        mock.info = info
        mock.balance_sheet = balance_sheet if balance_sheet is not None else pd.DataFrame()
        mock.cashflow = cashflow if cashflow is not None else pd.DataFrame()
        mock.financials = financials if financials is not None else pd.DataFrame()
        return mock

    def test_gibt_nur_befund_objekte_zurueck(self):
        with patch("agenten.agent1_bewertung.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker(
                {"marketCap": 100e9, "totalDebt": 10e9, "priceToBook": 1.5},
                _make_balance_sheet(),
                _make_cashflow(),
                _make_financials(),
            )
            befunde = analyse("TEST")
        for b in befunde:
            assert isinstance(b, Befund), "CLAUDE.md: kein Agent gibt nackte Zahl zurück"

    def test_leere_info_alles_unbestimmt_oder_nicht_passiert(self):
        with patch("agenten.agent1_bewertung.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker({})
            befunde = analyse("TEST")
        for b in befunde:
            assert b.zustand != Zustand.PASSIERT, "Fehlende Daten → nie PASSIERT"

    def test_exception_gibt_unbestimmt(self):
        with patch("agenten.agent1_bewertung.yf.Ticker") as mock_cls:
            mock_cls.side_effect = RuntimeError("Netz")
            befunde = analyse("TEST")
        assert all(b.bestanden is None for b in befunde)

    def test_vollstaendige_daten_bewertbar(self):
        info = {
            "marketCap": 100e9,
            "totalDebt": 10e9,
            "priceToBook": 1.5,
            "sector": "Technology",
        }
        with patch("agenten.agent1_bewertung.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker(
                info,
                _make_balance_sheet(),
                _make_cashflow(5e9),
                _make_financials(ebit=8e9, net_income=6e9, interest_expense=-500e6),
            )
            befunde = analyse("TEST")
        bewertbar = [b for b in befunde if b.bestanden is not None]
        assert len(bewertbar) >= 3, "Mindestens 3 Kriterien müssen bewertbar sein"

    # --- Regressionstests (aus vorheriger Session) ---

    def test_schulden_nicht_unbestimmt_wenn_equity_im_balance_sheet(self):
        """Bug-fix: totalStockholderEquity=None in info, aber Wert im balance_sheet vorhanden."""
        info = {
            "marketCap": 1e12,
            "totalDebt": 50e9,
            "totalStockholderEquity": None,
        }
        with patch("agenten.agent1_bewertung.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker(
                info,
                _make_balance_sheet(equity=200e9),
                _make_cashflow(5e9),
                _make_financials(),
            )
            befunde = analyse("TEST")
        schulden_b = next(b for b in befunde if "Schulden" in b.label)
        assert schulden_b.bestanden is not None, "Schulden/EK muss bewertbar sein"

    def test_fcf_aus_cashflow_nicht_aus_info(self):
        """Bug-fix: info['freeCashflow'] liefert falschen Wert."""
        info = {
            "marketCap": 3e12,
            "freeCashflow": 1e9,  # falsch: 1B statt 70B
        }
        with patch("agenten.agent1_bewertung.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker(
                info,
                pd.DataFrame(),
                _make_cashflow(70e9),  # korrekt im cashflow statement
                _make_financials(ebit=80e9),
            )
            befunde = analyse("TEST")
        fcf_b = next((b for b in befunde if "FCF" in b.label), None)
        if fcf_b is not None and fcf_b.wert is not None:
            assert fcf_b.wert > 0.005, "FCF muss aus cashflow Statement kommen"


class TestConviction:
    def test_alle_bestanden_gibt_100(self):
        befunde = [Befund(label=f"X{i}", bestanden=True) for i in range(7)]
        assert conviction(befunde) == 100.0

    def test_keine_bestanden_gibt_0(self):
        befunde = [Befund(label=f"X{i}", bestanden=False) for i in range(7)]
        assert conviction(befunde) == 0.0

    def test_alle_unbestimmt_gibt_0(self):
        befunde = [Befund(label=f"X{i}", bestanden=None) for i in range(7)]
        assert conviction(befunde) == 0.0

    def test_unbestimmt_zaehlt_nicht_mit(self):
        befunde = [
            Befund(label="A", bestanden=True),
            Befund(label="B", bestanden=None),
        ]
        assert conviction(befunde) == 100.0

    def test_schwellen_kommen_aus_dict(self):
        assert "ev_ebit_max" in SCHWELLEN
        assert "fcf_ev_yield_min" in SCHWELLEN
        assert "ccr_min" in SCHWELLEN
        assert "roce_min" in SCHWELLEN
        assert "zinsdeckung_min" in SCHWELLEN
        assert "conviction_gewicht" in SCHWELLEN

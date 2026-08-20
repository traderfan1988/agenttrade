"""Tests für kern/cause_classifier – kein Netz-Zugriff."""
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from kern.cause_classifier import (
    EARNINGS_VETO_TAGE,
    SEKTOR_ETF,
    DipDiagnose,
    Ursache,
    _berechne_sektor_delta,
    _klassifiziere_headline,
    _news_ursache,
    _tage_bis_earnings,
    klassifiziere,
)


class TestKlassifiziereHeadline:
    def test_governance_legal_wird_erkannt(self):
        assert _klassifiziere_headline("CEO accused of fraud and insider trading") == Ursache.GOVERNANCE_LEGAL

    def test_regulatory_wird_erkannt(self):
        assert _klassifiziere_headline("Company faces antitrust investigation") == Ursache.REGULATORY

    def test_ma_wird_erkannt(self):
        assert _klassifiziere_headline("Company agrees to merger with rival") == Ursache.MA

    def test_guidance_lowered_wird_erkannt(self):
        assert _klassifiziere_headline("Management issues profit warning for Q3") == Ursache.GUIDANCE_LOWERED

    def test_guidance_intact_wird_erkannt(self):
        assert _klassifiziere_headline("Quarterly results beats expectations") == Ursache.GUIDANCE_INTACT

    def test_macro_wird_erkannt(self):
        assert _klassifiziere_headline("Federal reserve hikes interest rate 50 bps") == Ursache.MACRO

    def test_unbekannter_text_gibt_none(self):
        assert _klassifiziere_headline("Stock price moves higher on volume") is None

    def test_governance_hat_prioritaet_vor_macro(self):
        result = _klassifiziere_headline("SEC investigation into fraud amid market sell-off")
        assert result == Ursache.GOVERNANCE_LEGAL

    def test_gross_kleinschreibung_ignoriert(self):
        assert _klassifiziere_headline("FRAUD SCANDAL AT COMPANY") == Ursache.GOVERNANCE_LEGAL


class TestNewsUrsache:
    def _mock_ticker_mit_news(self, titles: list) -> MagicMock:
        t = MagicMock()
        t.news = [{"title": title, "summary": ""} for title in titles]
        return t

    def test_leere_news_gibt_unknown(self):
        t = MagicMock()
        t.news = []
        assert _news_ursache(t) == Ursache.UNKNOWN

    def test_none_news_gibt_unknown(self):
        t = MagicMock()
        t.news = None
        assert _news_ursache(t) == Ursache.UNKNOWN

    def test_mehrheitliche_macro_news(self):
        t = self._mock_ticker_mit_news([
            "Federal reserve hikes interest rate",
            "Interest rate fears shake market",
            "GDP slowdown concerns grow",
        ])
        assert _news_ursache(t) == Ursache.MACRO

    def test_governance_gewinnt_bei_einem_treffer(self):
        t = self._mock_ticker_mit_news([
            "SEC investigation into CEO fraud",
            "Market rebounds after sell-off",
            "Inflation data released",
        ])
        assert _news_ursache(t) == Ursache.GOVERNANCE_LEGAL

    def test_exception_bei_news_gibt_unknown(self):
        t = MagicMock()
        type(t).news = property(lambda self: (_ for _ in ()).throw(RuntimeError("API error")))
        result = _news_ursache(t)
        assert result == Ursache.UNKNOWN


class TestTagesBisEarnings:
    def _mock_ticker_mit_calendar(self, earnings_date) -> MagicMock:
        t = MagicMock()
        t.calendar = {"Earnings Date": [earnings_date]}
        return t

    def test_morgen_gibt_1(self):
        morgen = date.today() + timedelta(days=1)
        t = self._mock_ticker_mit_calendar(pd.Timestamp(morgen))
        result = _tage_bis_earnings(t)
        assert result == 1

    def test_heute_gibt_0(self):
        heute = date.today()
        t = self._mock_ticker_mit_calendar(pd.Timestamp(heute))
        result = _tage_bis_earnings(t)
        assert result == 0

    def test_vergangenes_datum_gibt_0(self):
        gestern = date.today() - timedelta(days=5)
        t = self._mock_ticker_mit_calendar(pd.Timestamp(gestern))
        result = _tage_bis_earnings(t)
        assert result == 0

    def test_weit_zukunft_gibt_positiven_wert(self):
        in_60_tagen = date.today() + timedelta(days=60)
        t = self._mock_ticker_mit_calendar(pd.Timestamp(in_60_tagen))
        result = _tage_bis_earnings(t)
        assert result == 60

    def test_kein_calendar_gibt_none(self):
        t = MagicMock()
        t.calendar = None
        assert _tage_bis_earnings(t) is None

    def test_leere_liste_gibt_none(self):
        t = MagicMock()
        t.calendar = {"Earnings Date": []}
        assert _tage_bis_earnings(t) is None

    def test_exception_gibt_none(self):
        t = MagicMock()
        t.calendar = {"falsch": "daten"}
        result = _tage_bis_earnings(t)
        assert result is None


class TestBerechneSektorDelta:
    def _mock_preis_df(self, start: float, ende: float) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=2, freq="B")
        return pd.DataFrame({"Close": [start, ende]}, index=idx)

    def test_positive_delta_wenn_ticker_besser(self):
        with patch("kern.cause_classifier.yf.download") as mock_dl:
            mock_dl.side_effect = [
                self._mock_preis_df(100, 110),  # ticker: +10%
                self._mock_preis_df(100, 105),  # etf: +5%
            ]
            delta, etf = _berechne_sektor_delta("AAPL", "Technology")
        assert delta is not None
        assert abs(delta - 0.05) < 1e-4  # +5% outperformance

    def test_negative_delta_wenn_ticker_schlechter(self):
        with patch("kern.cause_classifier.yf.download") as mock_dl:
            mock_dl.side_effect = [
                self._mock_preis_df(100, 80),   # ticker: -20%
                self._mock_preis_df(100, 95),   # etf: -5%
            ]
            delta, etf = _berechne_sektor_delta("AAPL", "Technology")
        assert delta is not None
        assert delta < 0  # underperformt Sektor

    def test_unbekannter_sektor_gibt_none(self):
        with patch("kern.cause_classifier.yf.download"):
            delta, etf = _berechne_sektor_delta("AAPL", "Unbekannter Sektor")
        assert delta is None
        assert etf == ""

    def test_sektor_etf_wird_zurueckgegeben(self):
        with patch("kern.cause_classifier.yf.download") as mock_dl:
            mock_dl.side_effect = [
                self._mock_preis_df(100, 110),
                self._mock_preis_df(100, 105),
            ]
            delta, etf = _berechne_sektor_delta("AAPL", "Technology")
        assert etf == "XLK"

    def test_leere_daten_geben_none(self):
        with patch("kern.cause_classifier.yf.download") as mock_dl:
            mock_dl.return_value = pd.DataFrame()
            delta, etf = _berechne_sektor_delta("AAPL", "Technology")
        assert delta is None


class TestSeKtorETFMapping:
    def test_alle_hauptsektoren_haben_etf(self):
        sektoren = [
            "Technology", "Financial Services", "Healthcare",
            "Consumer Cyclical", "Consumer Defensive", "Energy",
            "Real Estate", "Utilities", "Industrials",
        ]
        for s in sektoren:
            assert s in SEKTOR_ETF, f"Sektor '{s}' fehlt im SEKTOR_ETF-Mapping"


class TestVetoBedingungen:
    def test_governance_setzt_veto(self):
        with (
            patch("kern.cause_classifier.yf.Ticker") as mock_cls,
            patch("kern.cause_classifier._news_ursache", return_value=Ursache.GOVERNANCE_LEGAL),
            patch("kern.cause_classifier._berechne_sektor_delta", return_value=(None, "")),
            patch("kern.cause_classifier._tage_bis_earnings", return_value=30),
        ):
            mock_cls.return_value.info = {"sector": "Technology", "symbol": "TEST"}
            diag = klassifiziere("TEST")
        assert diag.veto is True
        assert diag.ursache == Ursache.GOVERNANCE_LEGAL

    def test_earnings_morgen_setzt_veto(self):
        with (
            patch("kern.cause_classifier.yf.Ticker") as mock_cls,
            patch("kern.cause_classifier._news_ursache", return_value=Ursache.MACRO),
            patch("kern.cause_classifier._berechne_sektor_delta", return_value=(None, "")),
            patch("kern.cause_classifier._tage_bis_earnings", return_value=2),
        ):
            mock_cls.return_value.info = {"sector": "Technology", "symbol": "TEST"}
            diag = klassifiziere("TEST")
        assert diag.veto is True
        assert EARNINGS_VETO_TAGE > 2

    def test_earnings_weit_entfernt_kein_veto(self):
        with (
            patch("kern.cause_classifier.yf.Ticker") as mock_cls,
            patch("kern.cause_classifier._news_ursache", return_value=Ursache.MACRO),
            patch("kern.cause_classifier._berechne_sektor_delta", return_value=(-0.05, "XLK")),
            patch("kern.cause_classifier._tage_bis_earnings", return_value=60),
        ):
            mock_cls.return_value.info = {"sector": "Technology", "symbol": "TEST"}
            diag = klassifiziere("TEST")
        assert diag.veto is False

    def test_klassifiziere_gibt_dipdiagnose_zurueck(self):
        with (
            patch("kern.cause_classifier.yf.Ticker") as mock_cls,
            patch("kern.cause_classifier._news_ursache", return_value=Ursache.UNKNOWN),
            patch("kern.cause_classifier._berechne_sektor_delta", return_value=(None, "")),
            patch("kern.cause_classifier._tage_bis_earnings", return_value=None),
        ):
            mock_cls.return_value.info = {}
            result = klassifiziere("TEST")
        assert isinstance(result, DipDiagnose)

    def test_exception_gibt_valides_objekt(self):
        with patch("kern.cause_classifier.yf.Ticker") as mock_cls:
            mock_cls.side_effect = RuntimeError("Netz")
            diag = klassifiziere("TEST")
        assert isinstance(diag, DipDiagnose)
        assert diag.veto is False

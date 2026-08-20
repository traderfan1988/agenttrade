"""Tests für Agent 5 – Insider/Sentiment. Mocked, kein Netz-Zugriff."""
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from kern.typen import Befund, Zustand
from agenten.agent5_insider import (
    SCHWELLEN,
    _analyst_konsens,
    _insider_netto,
    _institutionelle_beteiligung,
    _short_interest,
    analyse,
    conviction,
)


def _make_insider_df(eintraege: list) -> pd.DataFrame:
    """eintraege: list of (tage_ago, shares, typ) – typ z.B. 'Purchase', 'Sale'."""
    rows, idx = [], []
    for tage_ago, shares, typ in eintraege:
        idx.append(pd.Timestamp(date.today() - timedelta(days=tage_ago)))
        rows.append({"Shares": shares, "Value": abs(shares) * 100, "Insider Trading": typ})
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def _make_recommendations(
    strong_buy: int = 5,
    buy: int = 10,
    hold: int = 8,
    sell: int = 2,
    strong_sell: int = 0,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "period": "0m",
        "strongBuy": strong_buy,
        "buy": buy,
        "hold": hold,
        "sell": sell,
        "strongSell": strong_sell,
    }])


class TestInsiderNetto:
    def test_mehr_kaeufe_als_verkauefe_besteht(self):
        df = _make_insider_df([
            (10, 100_000, "Purchase"),
            (20, 50_000, "Sale"),
        ])
        b = _insider_netto(df)
        assert b.bestanden is True

    def test_mehr_verkauefe_besteht_nicht(self):
        df = _make_insider_df([
            (10, 10_000, "Purchase"),
            (20, 200_000, "Sale"),
        ])
        b = _insider_netto(df)
        assert b.bestanden is False

    def test_nur_alte_transaktionen_sind_unbestimmt(self):
        # Transaktionen vor mehr als 90 Tagen → außerhalb Fenster
        df = _make_insider_df([
            (120, 100_000, "Purchase"),
            (150, 50_000, "Sale"),
        ])
        b = _insider_netto(df)
        assert b.bestanden is None

    def test_leerer_df_ist_unbestimmt(self):
        b = _insider_netto(pd.DataFrame())
        assert b.bestanden is None
        assert b.zustand == Zustand.UNBESTIMMT

    def test_none_ist_unbestimmt(self):
        b = _insider_netto(None)
        assert b.bestanden is None

    def test_fehlende_daten_nie_passiert(self):
        b = _insider_netto(pd.DataFrame())
        assert b.zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        df = _make_insider_df([(5, 100_000, "Purchase")])
        assert isinstance(_insider_netto(df), Befund)

    def test_wert_ist_netto_aktien(self):
        df = _make_insider_df([
            (5, 100_000, "Purchase"),
            (10, 30_000, "Sale"),
        ])
        b = _insider_netto(df)
        assert b.wert == pytest.approx(70_000)


class TestShortInterest:
    def test_niedrig_besteht(self):
        b = _short_interest(SCHWELLEN["short_interest_max"] - 0.01)
        assert b.bestanden is True

    def test_hoch_besteht_nicht(self):
        b = _short_interest(SCHWELLEN["short_interest_max"] + 0.05)
        assert b.bestanden is False

    def test_none_ist_unbestimmt(self):
        b = _short_interest(None)
        assert b.bestanden is None
        assert b.zustand == Zustand.UNBESTIMMT

    def test_fehlende_daten_nie_passiert(self):
        assert _short_interest(None).zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        assert isinstance(_short_interest(0.05), Befund)

    def test_wert_ist_short_pct(self):
        b = _short_interest(0.07)
        assert b.wert == pytest.approx(0.07, rel=1e-4)


class TestAnalystKonsens:
    def test_mehrheit_buy_besteht(self):
        # 15 Buy/StrongBuy, 10 Hold/Sell → 60% Buy
        rec = _make_recommendations(strong_buy=5, buy=10, hold=8, sell=2, strong_sell=0)
        b = _analyst_konsens(rec)
        assert b.bestanden is True

    def test_mehrheit_hold_besteht_nicht(self):
        # 3 Buy, 20 Hold/Sell → 12% Buy
        rec = _make_recommendations(strong_buy=1, buy=2, hold=15, sell=4, strong_sell=1)
        b = _analyst_konsens(rec)
        assert b.bestanden is False

    def test_leerer_df_ist_unbestimmt(self):
        b = _analyst_konsens(pd.DataFrame())
        assert b.bestanden is None

    def test_none_ist_unbestimmt(self):
        b = _analyst_konsens(None)
        assert b.bestanden is None

    def test_alle_null_ist_unbestimmt(self):
        rec = _make_recommendations(0, 0, 0, 0, 0)
        b = _analyst_konsens(rec)
        assert b.bestanden is None

    def test_fehlende_daten_nie_passiert(self):
        assert _analyst_konsens(None).zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        rec = _make_recommendations()
        assert isinstance(_analyst_konsens(rec), Befund)

    def test_wert_ist_buy_ratio(self):
        # 10 StrongBuy+Buy, 10 Rest → 50%
        rec = _make_recommendations(strong_buy=5, buy=5, hold=5, sell=3, strong_sell=2)
        b = _analyst_konsens(rec)
        assert b.wert == pytest.approx(0.50, rel=1e-4)


class TestInstitutionelleBeteiligung:
    def test_im_guenstigen_bereich_besteht(self):
        b = _institutionelle_beteiligung(0.70)  # 70% → OK
        assert b.bestanden is True

    def test_zu_niedrig_besteht_nicht(self):
        b = _institutionelle_beteiligung(0.10)  # < 30%
        assert b.bestanden is False

    def test_zu_hoch_besteht_nicht(self):
        b = _institutionelle_beteiligung(0.99)  # > 98% → kein Streubesitz
        assert b.bestanden is False

    def test_none_ist_unbestimmt(self):
        b = _institutionelle_beteiligung(None)
        assert b.bestanden is None
        assert b.zustand == Zustand.UNBESTIMMT

    def test_fehlende_daten_nie_passiert(self):
        assert _institutionelle_beteiligung(None).zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        assert isinstance(_institutionelle_beteiligung(0.65), Befund)

    def test_grenzwert_genau_mindest_besteht(self):
        b = _institutionelle_beteiligung(SCHWELLEN["institutionell_min"])
        assert b.bestanden is True


class TestAnalyse:
    def _mock_ticker(self, info: dict, insider_txn, recommendations):
        mock = MagicMock()
        mock.info = info
        mock.insider_transactions = insider_txn
        mock.recommendations = recommendations
        return mock

    def test_gibt_nur_befund_objekte_zurueck(self):
        insider = _make_insider_df([(10, 100_000, "Purchase")])
        rec = _make_recommendations()
        info = {"shortPercentOfFloat": 0.05, "heldPercentInstitutions": 0.70}
        with patch("agenten.agent5_insider.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker(info, insider, rec)
            befunde = analyse("TEST")
        for b in befunde:
            assert isinstance(b, Befund), "CLAUDE.md: kein Agent gibt nackte Zahl zurück"

    def test_leere_daten_geben_unbestimmt(self):
        with patch("agenten.agent5_insider.yf.Ticker") as mock_cls:
            mock_cls.return_value = self._mock_ticker({}, pd.DataFrame(), pd.DataFrame())
            befunde = analyse("TEST")
        for b in befunde:
            assert b.zustand != Zustand.PASSIERT

    def test_exception_gibt_unbestimmt(self):
        with patch("agenten.agent5_insider.yf.Ticker") as mock_cls:
            mock_cls.side_effect = RuntimeError("Netz")
            befunde = analyse("TEST")
        assert all(b.bestanden is None for b in befunde)

    def test_fehlende_daten_nie_passiert(self):
        with patch("agenten.agent5_insider.yf.Ticker") as mock_cls:
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
        befunde = [Befund(label="X", bestanden=None)]
        assert conviction(befunde) == 0.0

    def test_haelfte_gibt_50(self):
        befunde = [Befund(label="A", bestanden=True), Befund(label="B", bestanden=False)]
        assert conviction(befunde) == 50.0

    def test_schwellen_kommen_aus_dict(self):
        assert "short_interest_max" in SCHWELLEN
        assert "analyst_buy_ratio_min" in SCHWELLEN
        assert "institutionell_min" in SCHWELLEN
        assert "conviction_gewicht" in SCHWELLEN

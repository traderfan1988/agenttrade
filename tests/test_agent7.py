"""Tests für Agent 7 – Momentum. Mocked, kein Netz-Zugriff."""
from unittest.mock import patch

import pandas as pd
import pytest

from kern.typen import Befund, Zustand
from agenten.agent7_momentum import (
    SCHWELLEN,
    _ma200_abstand,
    _momentum_12m,
    _relative_staerke_6m,
    _rsi,
    _rsi_zone,
    analyse,
    conviction,
)


def _preisserie(werte: list, start="2020-01-01") -> pd.Series:
    idx = pd.date_range(start=start, periods=len(werte), freq="B")
    return pd.Series(werte, index=idx, dtype=float)


def _steigende_serie(n=60, start=100.0, delta=1.0) -> pd.Series:
    vals = [start + i * delta for i in range(n)]
    return _preisserie(vals)


def _fallende_serie(n=60, start=160.0, delta=1.0) -> pd.Series:
    vals = [start - i * delta for i in range(n)]
    return _preisserie(vals)


def _spy_mock(n=130, ret=0.0) -> pd.DataFrame:
    """Synthetischer SPY-DataFrame für yf.download-Rückgabe."""
    start = 400.0
    vals = [start * (1 + ret) ** (i / 252) for i in range(n)]
    idx = pd.date_range(end="2024-12-31", periods=n, freq="B")
    return pd.DataFrame({"Close": vals}, index=idx)


def _ticker_mock(n=260, ret=0.0) -> pd.DataFrame:
    start = 100.0
    vals = [start * (1 + ret) ** (i / 252) for i in range(n)]
    idx = pd.date_range(end="2024-12-31", periods=n, freq="B")
    return pd.DataFrame({"Close": vals}, index=idx)


class TestRsiHelper:
    def test_rein_steigende_preise_geben_hohen_rsi(self):
        preise = _steigende_serie(n=60)
        r = _rsi(preise)
        assert r is not None and r > 70

    def test_rein_fallende_preise_geben_niedrigen_rsi(self):
        preise = _fallende_serie(n=60)
        r = _rsi(preise)
        assert r is not None and r < 30

    def test_zu_kurze_serie_gibt_none(self):
        preise = _preisserie([100.0, 101.0])
        assert _rsi(preise) is None

    def test_ergebnis_zwischen_0_und_100(self):
        preise = _steigende_serie(n=60)
        r = _rsi(preise)
        assert r is not None
        assert 0 <= r <= 100


class TestMomentum12m:
    def test_positive_rendite_besteht(self):
        preise = _steigende_serie(n=260)
        b = _momentum_12m(preise)
        assert b.bestanden is True

    def test_tiefer_absturz_besteht_nicht(self):
        # Fällt weit mehr als 15%
        vals = [100.0 - i * 0.5 for i in range(260)]
        b = _momentum_12m(_preisserie(vals))
        assert b.bestanden is False

    def test_zu_wenig_daten_ist_unbestimmt(self):
        preise = _steigende_serie(n=10)
        b = _momentum_12m(preise)
        assert b.bestanden is None

    def test_gibt_befund_zurueck(self):
        assert isinstance(_momentum_12m(_steigende_serie()), Befund)

    def test_fehlende_daten_nie_passiert(self):
        assert _momentum_12m(pd.Series(dtype=float)).zustand != Zustand.PASSIERT


class TestRelativStaerke6m:
    def test_outperformt_besteht(self):
        # Ticker +20%, SPY +5%
        ticker = _ticker_mock(n=130, ret=0.20)
        spy = _spy_mock(n=130, ret=0.05)
        b = _relative_staerke_6m(ticker["Close"], spy["Close"])
        assert b.bestanden is True

    def test_stark_underperformt_besteht_nicht(self):
        # Ticker -30%, SPY +5%
        ticker = _ticker_mock(n=130, ret=-0.30)
        spy = _spy_mock(n=130, ret=0.05)
        b = _relative_staerke_6m(ticker["Close"], spy["Close"])
        assert b.bestanden is False

    def test_leere_spy_daten_sind_unbestimmt(self):
        ticker = _ticker_mock(n=130)
        b = _relative_staerke_6m(ticker["Close"], pd.Series(dtype=float))
        assert b.bestanden is None

    def test_leere_ticker_daten_sind_unbestimmt(self):
        spy = _spy_mock(n=130)
        b = _relative_staerke_6m(pd.Series(dtype=float), spy["Close"])
        assert b.bestanden is None

    def test_gibt_befund_zurueck(self):
        ticker = _ticker_mock(n=130)
        spy = _spy_mock(n=130)
        assert isinstance(_relative_staerke_6m(ticker["Close"], spy["Close"]), Befund)

    def test_fehlende_daten_nie_passiert(self):
        b = _relative_staerke_6m(pd.Series(dtype=float), pd.Series(dtype=float))
        assert b.zustand != Zustand.PASSIERT


class TestRsiZone:
    def test_in_der_zone_besteht(self):
        rsi_mid = (SCHWELLEN["rsi_min"] + SCHWELLEN["rsi_max"]) / 2
        b = _rsi_zone(rsi_mid)
        assert b.bestanden is True

    def test_ueberkauft_besteht_nicht(self):
        b = _rsi_zone(SCHWELLEN["rsi_max"] + 5)
        assert b.bestanden is False

    def test_panisch_ausverkauft_besteht_nicht(self):
        b = _rsi_zone(SCHWELLEN["rsi_min"] - 5)
        assert b.bestanden is False

    def test_none_ist_unbestimmt(self):
        b = _rsi_zone(None)
        assert b.bestanden is None

    def test_fehlende_daten_nie_passiert(self):
        assert _rsi_zone(None).zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        assert isinstance(_rsi_zone(50.0), Befund)

    def test_untere_grenze_besteht(self):
        b = _rsi_zone(SCHWELLEN["rsi_min"])
        assert b.bestanden is True

    def test_obere_grenze_besteht(self):
        b = _rsi_zone(SCHWELLEN["rsi_max"])
        assert b.bestanden is True


class TestMa200Abstand:
    def test_nah_an_ma200_besteht(self):
        # Preise steigen leicht → nahe am MA200
        preise = _steigende_serie(n=250, delta=0.1)
        b = _ma200_abstand(preise)
        assert b.bestanden is True

    def test_weit_ueber_ma200_besteht_nicht(self):
        # Erste 200 Tage flach, letzter Tag ×5 → weit über MA
        vals = [100.0] * 249 + [600.0]
        b = _ma200_abstand(_preisserie(vals))
        assert b.bestanden is False

    def test_zu_wenig_daten_ist_unbestimmt(self):
        b = _ma200_abstand(_steigende_serie(n=10))
        assert b.bestanden is None

    def test_gibt_befund_zurueck(self):
        assert isinstance(_ma200_abstand(_steigende_serie(n=250)), Befund)

    def test_fehlende_daten_nie_passiert(self):
        assert _ma200_abstand(pd.Series(dtype=float)).zustand != Zustand.PASSIERT


class TestAnalyse:
    def _download_side_effect(self, ticker_data, spy_data):
        def side(ticker, **kw):
            if ticker == "SPY":
                return spy_data
            return ticker_data
        return side

    def test_gibt_nur_befund_objekte_zurueck(self):
        ticker_df = _ticker_mock(n=270)
        spy_df = _spy_mock(n=270)
        with patch("agenten.agent7_momentum.yf.download") as mock_dl:
            mock_dl.side_effect = self._download_side_effect(ticker_df, spy_df)
            befunde = analyse("TEST")
        for b in befunde:
            assert isinstance(b, Befund)

    def test_leere_daten_geben_unbestimmt(self):
        with patch("agenten.agent7_momentum.yf.download") as mock_dl:
            mock_dl.return_value = pd.DataFrame()
            befunde = analyse("TEST")
        for b in befunde:
            assert b.zustand != Zustand.PASSIERT

    def test_exception_gibt_unbestimmt(self):
        with patch("agenten.agent7_momentum.yf.download") as mock_dl:
            mock_dl.side_effect = RuntimeError("Netz")
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
        assert "rsi_min" in SCHWELLEN
        assert "rsi_max" in SCHWELLEN
        assert "momentum_12m_min" in SCHWELLEN
        assert "conviction_gewicht" in SCHWELLEN

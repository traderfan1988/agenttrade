"""Tests für Agent 3 – Dip-Diagnose. Mocked, kein Netz-Zugriff."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from kern.cause_classifier import DipDiagnose, Ursache
from kern.typen import Befund, Zustand
from agenten.agent3_dip import (
    SCHWELLEN,
    _aktueller_drawdown,
    _befund_drawdown_vs_hist,
    _befund_earnings_veto,
    _befund_sektor_relativ,
    _befund_ursache,
    _hist_drawdown_perzentil,
    analyse,
    conviction,
)


def _serie(werte: list) -> pd.Series:
    idx = pd.date_range("2019-01-01", periods=len(werte), freq="B")
    return pd.Series(werte, index=idx, dtype=float)


def _diag(
    ursache: Ursache = Ursache.MACRO,
    sektor_delta: float = -0.10,
    tage_bis_earnings: int = 60,
    veto: bool = False,
    veto_grund: str = "",
    sektor_etf: str = "XLK",
) -> DipDiagnose:
    return DipDiagnose(
        ursache=ursache,
        sektor_delta=sektor_delta,
        tage_bis_earnings=tage_bis_earnings,
        veto=veto,
        veto_grund=veto_grund,
        sektor_etf=sektor_etf,
    )


class TestHistDrawdownPerzentil:
    def test_ausreichend_daten_gibt_float(self):
        # 300 Tage: hoch → runter → hoch Muster
        werte = [100.0] * 100 + [80.0] * 100 + [100.0] * 100
        s = _serie(werte)
        result = _hist_drawdown_perzentil(s, 80)
        assert result is not None
        assert isinstance(result, float)

    def test_zu_wenig_daten_gibt_none(self):
        s = _serie([100.0] * 200)  # < 252
        assert _hist_drawdown_perzentil(s, 80) is None

    def test_immer_positiv(self):
        werte = [100.0] * 100 + [70.0] * 100 + [90.0] * 100
        s = _serie(werte)
        result = _hist_drawdown_perzentil(s, 80)
        assert result is not None
        assert result >= 0

    def test_hoeheres_perzentil_groesser_gleich_niedrigerem(self):
        werte = list(range(50, 350))  # steigend mit Volatilität
        werte += [100.0] * 100
        s = _serie(werte)
        p50 = _hist_drawdown_perzentil(s, 50)
        p80 = _hist_drawdown_perzentil(s, 80)
        if p50 is not None and p80 is not None:
            assert p80 >= p50

    def test_kein_drawdown_in_steter_aufwaertsserie(self):
        werte = list(range(100, 100 + 500))  # monoton steigend
        s = _serie(werte)
        result = _hist_drawdown_perzentil(s, 80)
        if result is not None:
            assert result == pytest.approx(0.0, abs=1e-6)


class TestAktuellerDrawdown:
    def test_korrekte_berechnung(self):
        s = _serie([100.0] * 200 + [80.0] * 52)
        dd = _aktueller_drawdown(s)
        assert dd is not None
        assert abs(dd - 0.20) < 1e-6

    def test_leere_serie_gibt_none(self):
        assert _aktueller_drawdown(pd.Series(dtype=float)) is None

    def test_immer_positiv(self):
        s = _serie([80.0, 90.0, 100.0, 85.0, 70.0, 75.0])
        dd = _aktueller_drawdown(s)
        assert dd is not None
        assert dd >= 0


class TestBefundDrawdownVsHist:
    def test_ueber_perzentil_besteht(self):
        b = _befund_drawdown_vs_hist(0.30, 0.20)
        assert b.bestanden is True
        assert b.zustand == Zustand.PASSIERT

    def test_unter_perzentil_besteht_nicht(self):
        b = _befund_drawdown_vs_hist(0.10, 0.25)
        assert b.bestanden is False
        assert b.zustand == Zustand.NICHT_PASSIERT

    def test_zu_kleiner_drawdown_besteht_nicht(self):
        b = _befund_drawdown_vs_hist(0.03, 0.20)  # DD < drawdown_min
        assert b.bestanden is False

    def test_extremer_drawdown_besteht_nicht(self):
        b = _befund_drawdown_vs_hist(0.70, 0.20)  # DD > drawdown_extrem_max
        assert b.bestanden is False

    def test_none_drawdown_ist_unbestimmt(self):
        b = _befund_drawdown_vs_hist(None, 0.20)
        assert b.bestanden is None
        assert b.zustand == Zustand.UNBESTIMMT

    def test_none_hist_ist_unbestimmt(self):
        b = _befund_drawdown_vs_hist(0.20, None)
        assert b.bestanden is None

    def test_gibt_befund_zurueck(self):
        assert isinstance(_befund_drawdown_vs_hist(0.25, 0.20), Befund)


class TestBefundUrsache:
    def test_macro_ist_investierbar(self):
        b = _befund_ursache(_diag(ursache=Ursache.MACRO))
        assert b.bestanden is True

    def test_sector_ist_investierbar(self):
        b = _befund_ursache(_diag(ursache=Ursache.SECTOR))
        assert b.bestanden is True

    def test_guidance_intact_ist_investierbar(self):
        b = _befund_ursache(_diag(ursache=Ursache.GUIDANCE_INTACT))
        assert b.bestanden is True

    def test_unknown_gilt_als_investierbar(self):
        b = _befund_ursache(_diag(ursache=Ursache.UNKNOWN))
        assert b.bestanden is True

    def test_governance_legal_ist_nicht_investierbar(self):
        b = _befund_ursache(_diag(ursache=Ursache.GOVERNANCE_LEGAL, veto=True, veto_grund="Governance"))
        assert b.bestanden is False
        assert b.zustand == Zustand.NICHT_PASSIERT

    def test_guidance_lowered_ist_nicht_investierbar(self):
        b = _befund_ursache(_diag(ursache=Ursache.GUIDANCE_LOWERED))
        assert b.bestanden is False

    def test_regulatory_ist_nicht_investierbar(self):
        b = _befund_ursache(_diag(ursache=Ursache.REGULATORY))
        assert b.bestanden is False

    def test_gibt_befund_zurueck(self):
        assert isinstance(_befund_ursache(_diag()), Befund)

    def test_wert_enthaelt_ursache_string(self):
        b = _befund_ursache(_diag(ursache=Ursache.MACRO))
        assert b.wert == Ursache.MACRO.value


class TestBefundEarningsVeto:
    def test_weit_entfernte_earnings_bestehen(self):
        b = _befund_earnings_veto(_diag(tage_bis_earnings=60))
        assert b.bestanden is True

    def test_nahe_earnings_bestehen_nicht(self):
        b = _befund_earnings_veto(_diag(tage_bis_earnings=2, veto=True, veto_grund="Earnings in 2 Tag(en)"))
        assert b.bestanden is False

    def test_unbekannte_earnings_sind_unbestimmt(self):
        b = _befund_earnings_veto(_diag(tage_bis_earnings=None))
        assert b.bestanden is None
        assert b.zustand == Zustand.UNBESTIMMT

    def test_gibt_befund_zurueck(self):
        assert isinstance(_befund_earnings_veto(_diag()), Befund)

    def test_fehlende_daten_nie_passiert(self):
        b = _befund_earnings_veto(_diag(tage_bis_earnings=None))
        assert b.zustand != Zustand.PASSIERT


class TestBefundSektorRelativ:
    def test_sektor_dip_besteht_wenn_delta_klein(self):
        b = _befund_sektor_relativ(_diag(sektor_delta=-0.02))  # tracking sector
        assert b.bestanden is True

    def test_starkes_underperforming_besteht_nicht(self):
        b = _befund_sektor_relativ(_diag(sektor_delta=-0.10))
        assert b.bestanden is False

    def test_extremes_underperforming_besteht_nicht(self):
        b = _befund_sektor_relativ(_diag(sektor_delta=-0.50))
        assert b.bestanden is False

    def test_outperformance_besteht(self):
        b = _befund_sektor_relativ(_diag(sektor_delta=0.05))
        assert b.bestanden is True

    def test_none_delta_ist_unbestimmt(self):
        b = _befund_sektor_relativ(_diag(sektor_delta=None))
        assert b.bestanden is None
        assert b.zustand == Zustand.UNBESTIMMT

    def test_fehlende_daten_nie_passiert(self):
        b = _befund_sektor_relativ(_diag(sektor_delta=None))
        assert b.zustand != Zustand.PASSIERT

    def test_gibt_befund_zurueck(self):
        assert isinstance(_befund_sektor_relativ(_diag()), Befund)


class TestAnalyse:
    def _make_preis_df(self, werte: list) -> pd.DataFrame:
        idx = pd.date_range("2019-01-01", periods=len(werte), freq="B")
        return pd.DataFrame({"Close": werte}, index=idx)

    def test_gibt_nur_befund_objekte_zurueck(self):
        werte = [100.0] * 800 + [75.0] * 500
        with (
            patch("agenten.agent3_dip.yf.download") as mock_dl,
            patch("agenten.agent3_dip.klassifiziere") as mock_klass,
        ):
            mock_dl.return_value = self._make_preis_df(werte)
            mock_klass.return_value = _diag()
            befunde = analyse("TEST")
        for b in befunde:
            assert isinstance(b, Befund), "CLAUDE.md: kein Agent gibt nackte Zahl zurück"

    def test_leere_daten_geben_unbestimmt(self):
        with patch("agenten.agent3_dip.yf.download") as mock_dl:
            mock_dl.return_value = pd.DataFrame()
            befunde = analyse("TEST")
        assert all(b.bestanden is None for b in befunde)

    def test_exception_gibt_unbestimmt(self):
        with patch("agenten.agent3_dip.yf.download") as mock_dl:
            mock_dl.side_effect = RuntimeError("Netz")
            befunde = analyse("TEST")
        assert all(b.bestanden is None for b in befunde)

    def test_fehlende_daten_nie_passiert(self):
        with patch("agenten.agent3_dip.yf.download") as mock_dl:
            mock_dl.return_value = pd.DataFrame()
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
        assert "drawdown_min" in SCHWELLEN
        assert "hist_perzentil" in SCHWELLEN
        assert "sektor_delta_idio_min" in SCHWELLEN
        assert "conviction_gewicht" in SCHWELLEN

"""
Agent 3 – Dip-Diagnose.

Beantwortet: Ist der aktuelle Rückgang ein investierbarer Einstiegspunkt?

1. Drawdown-Niveau vs. hist. 80. Perzentil (5 Jahre rolling)
2. Ursachen-Bewertung (investierbar vs. Veto via cause_classifier)
3. Earnings-Proximity-Check (Veto wenn < 5 Tage)
4. Sektor-Relativperformance (Sektor-Dip vs. idiosynkratisch)
"""
from typing import List, Optional

import pandas as pd
import yfinance as yf

from kern.cause_classifier import DipDiagnose, Ursache, klassifiziere
from kern.typen import Befund, Zustand, befund_unbestimmt

SCHWELLEN = {
    "drawdown_min": 0.10,             # mind. 10% Rückgang nötig
    "drawdown_extrem_max": 0.60,      # > 60% → Value Trap Warnung
    "hist_perzentil": 80,             # 80. Perzentil der histor. Drawdowns
    "hist_fenster_jahre": 5,          # 5 Jahre Preishistorie
    "sektor_delta_idio_min": 0.05,    # ab 5% Underperformance = idiosynkratisch
    "sektor_delta_extrem": 0.40,      # ab 40% Underperformance = Strukturproblem-Warnung
    "conviction_gewicht": 0.333,
}

_INVESTIERBARE_URSACHEN = {
    Ursache.MACRO,
    Ursache.SECTOR,
    Ursache.GUIDANCE_INTACT,
    Ursache.UNKNOWN,
}


def _hist_drawdown_perzentil(preise: pd.Series, perzentil: int = 80) -> Optional[float]:
    """80. Perzentil der rollenden 252-Tage-Drawdowns über die gesamte Preisserie."""
    if len(preise) < 252:
        return None

    rolling_max = preise.rolling(window=252, min_periods=252).max()
    drawdowns = ((rolling_max - preise) / rolling_max).dropna()
    if drawdowns.empty:
        return None
    return float(drawdowns.quantile(perzentil / 100))


def _aktueller_drawdown(preise: pd.Series) -> Optional[float]:
    """Aktueller Drawdown vom 52W-Hoch (letzte 252 Handelstage)."""
    if preise.empty:
        return None
    fenster = preise.tail(252)
    hoch = float(fenster.max())
    if hoch <= 0:
        return None
    aktuell = float(fenster.iloc[-1])
    return (hoch - aktuell) / hoch


def _befund_drawdown_vs_hist(akt_dd: Optional[float], hist_p: Optional[float]) -> Befund:
    """Aktueller Drawdown vs. historischen 80. Perzentil – günstiger Einstieg?"""
    label = "Drawdown vs. hist. 80. Perzentil (5J)"
    if akt_dd is None or hist_p is None:
        return befund_unbestimmt(label, "Nicht genug Preishistorie (mind. 5 Jahre nötig)")

    if akt_dd > SCHWELLEN["drawdown_extrem_max"]:
        return Befund(
            label=label, wert=round(akt_dd, 4), bestanden=False,
            zustand=Zustand.NICHT_PASSIERT,
            details=f"Extremer DD {akt_dd:.1%} > {SCHWELLEN['drawdown_extrem_max']:.0%} – Value Trap prüfen",
        )

    if akt_dd < SCHWELLEN["drawdown_min"]:
        return Befund(
            label=label, wert=round(akt_dd, 4), bestanden=False,
            zustand=Zustand.NICHT_PASSIERT,
            details=f"DD {akt_dd:.1%} zu gering (Schwelle ≥ {SCHWELLEN['drawdown_min']:.0%})",
        )

    bestanden = akt_dd >= hist_p
    return Befund(
        label=label, wert=round(akt_dd, 4), bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=(
            f"Akt. DD {akt_dd:.1%} ≥ hist. P{SCHWELLEN['hist_perzentil']} {hist_p:.1%} – günstiger Einstiegspunkt"
            if bestanden else
            f"Akt. DD {akt_dd:.1%} < hist. P{SCHWELLEN['hist_perzentil']} {hist_p:.1%} – kein Extrempunkt"
        ),
    )


def _befund_ursache(diagnose: DipDiagnose) -> Befund:
    """Ist die klassifizierte Rückgangsursache für einen Value-Einstieg investierbar?"""
    label = "Ursachen-Bewertung"

    if diagnose.veto and diagnose.ursache not in _INVESTIERBARE_URSACHEN:
        return Befund(
            label=label, wert=diagnose.ursache.value, bestanden=False,
            zustand=Zustand.NICHT_PASSIERT,
            details=f"Hartes Veto: {diagnose.veto_grund} [{diagnose.ursache.value}]",
        )

    investierbar = diagnose.ursache in _INVESTIERBARE_URSACHEN
    return Befund(
        label=label, wert=diagnose.ursache.value, bestanden=investierbar,
        zustand=Zustand.PASSIERT if investierbar else Zustand.NICHT_PASSIERT,
        details=f"Ursache: {diagnose.ursache.value}",
    )


def _befund_earnings_veto(diagnose: DipDiagnose) -> Befund:
    """Earnings-Proximity: Kein Einstieg wenn Quartalsbericht in < 5 Tagen."""
    label = "Earnings-Proximity"
    tage = diagnose.tage_bis_earnings

    if tage is None:
        return befund_unbestimmt(label, "Earnings-Termin unbekannt")

    if diagnose.veto and "Earnings" in diagnose.veto_grund:
        return Befund(
            label=label, wert=tage, bestanden=False,
            zustand=Zustand.NICHT_PASSIERT,
            details=f"Veto: Earnings in {tage} Tag(en) – Unsicherheit zu hoch",
        )

    return Befund(
        label=label, wert=tage, bestanden=True,
        zustand=Zustand.PASSIERT,
        details=f"Kein zeitliches Veto: Earnings in {tage} Tagen",
    )


def _befund_sektor_relativ(diagnose: DipDiagnose) -> Befund:
    """Sektor-Relativperformance: Sektor-Dip (investierbar) oder idiosynkratisch (Warnsignal)?

    delta = Ticker-Rendite - ETF-Rendite.
    Nahe 0 = Stock bewegt sich mit Sektor (Sektor-Dip) → PASSIERT.
    Stark negativ = Stock unterperformt Sektor stark (company-spezifisch) → NICHT_PASSIERT.
    """
    label = "Sektor-Relativperformance"
    delta = diagnose.sektor_delta

    if delta is None:
        return befund_unbestimmt(label, "Kein Sektor-ETF-Vergleich verfügbar")

    if delta <= -SCHWELLEN["sektor_delta_extrem"]:
        return Befund(
            label=label, wert=round(delta, 4), bestanden=False,
            zustand=Zustand.NICHT_PASSIERT,
            details=(
                f"Extremes Underperforming ({delta:+.1%} vs. {diagnose.sektor_etf}) – "
                "Strukturproblem möglich"
            ),
        )

    if delta <= -SCHWELLEN["sektor_delta_idio_min"]:
        return Befund(
            label=label, wert=round(delta, 4), bestanden=False,
            zustand=Zustand.NICHT_PASSIERT,
            details=f"Unterperformt Sektor-ETF {diagnose.sektor_etf} ({delta:+.1%}) – Ursache prüfen",
        )

    return Befund(
        label=label, wert=round(delta, 4), bestanden=True,
        zustand=Zustand.PASSIERT,
        details=f"Sektor-Dip: Stock hält mit ETF {diagnose.sektor_etf} ({delta:+.1%})",
    )


def analyse(ticker: str) -> List[Befund]:
    """Dip-Diagnose. Gibt immer 4 Befund-Objekte zurück."""
    try:
        jahre = SCHWELLEN["hist_fenster_jahre"]
        daten = yf.download(ticker, period=f"{jahre}y", progress=False, auto_adjust=True)
    except Exception as exc:
        return [befund_unbestimmt(f"Dip-Diagnose {ticker}", f"Datenabruf: {exc}")]

    if daten.empty or "Close" not in daten.columns:
        return [befund_unbestimmt(f"Dip-Diagnose {ticker}", "Keine Kursdaten")]

    close = daten["Close"].squeeze()
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    close = close.dropna()

    akt_dd = _aktueller_drawdown(close)
    hist_p = _hist_drawdown_perzentil(close, SCHWELLEN["hist_perzentil"])

    try:
        diagnose = klassifiziere(ticker, fenster_tage=30)
    except Exception as exc:
        diagnose = DipDiagnose(
            ursache=Ursache.UNKNOWN,
            sektor_delta=None,
            tage_bis_earnings=None,
            veto=False,
            veto_grund=f"Klassifikation fehlgeschlagen: {exc}",
        )

    return [
        _befund_drawdown_vs_hist(akt_dd, hist_p),
        _befund_ursache(diagnose),
        _befund_earnings_veto(diagnose),
        _befund_sektor_relativ(diagnose),
    ]


def conviction(befunde: List[Befund]) -> float:
    """Score 0–100: Anteil bestandener, bewertbarer Kriterien × 100."""
    bewertbar = [b for b in befunde if b.bestanden is not None]
    if not bewertbar:
        return 0.0
    return round(sum(1 for b in bewertbar if b.bestanden) / len(bewertbar) * 100, 1)

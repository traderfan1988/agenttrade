"""
Agent 2 – Drawdown-Analyse, verankert am 52W-Hoch.

Kernregel: Vergleichsbasis ist immer Drawdown gegen Drawdown im selben Fenster.
Das Fenster startet am 52W-Hoch der Aktie.
Renditen im selben Fenster werden NICHT verglichen (→ systematischer Bias,
da das Fenster per Definition am Hoch beginnt und Renditen daraus immer ≤ 0).
"""
from typing import List, Optional

import pandas as pd
import yfinance as yf

from kern.typen import Befund, Zustand, befund_unbestimmt

SCHWELLEN = {
    "drawdown_guenstig_min": 0.10,      # mind. 10% unter 52W-Hoch = interessant
    "drawdown_extrem_max": 0.60,        # > 60% = mögliche Value Trap
    "fenster_tage": 252,                # 52W = ~252 Handelstage
    "naehe_tief_faktor": 0.70,          # Akt. DD ≥ 70% des Max-DD im Fenster
    "hist_vergleich_faktor": 0.70,      # Akt. DD ≥ 70% des Vorjahres-DD
    "dauer_min_tage": 20,               # mind. 20 Handelstage seit 52W-Hoch
    "dauer_max_tage": 400,              # > 400 Tage → mögliche Value Trap
    "conviction_gewicht": 0.333,        # 3 Agenten → gleiches Gewicht
}


def _drawdown_aktuell(preise: pd.Series) -> Optional[float]:
    """Aktueller Drawdown vom Maximum der Serie (immer ≥ 0)."""
    if preise.empty:
        return None
    hoch = float(preise.max())
    if hoch <= 0:
        return None
    aktuell = float(preise.iloc[-1])
    return (hoch - aktuell) / hoch


def _max_dd_im_fenster(fenster: pd.Series) -> Optional[float]:
    """Maximaler historischer Drawdown im Fenster, verankert am ersten Wert (Hoch).

    Fenster startet am 52W-Hoch → erster Wert ist das Hoch.
    Kein Rendite-Bias: wir messen nur Tiefe des Rückgangs vom Hoch.
    """
    if len(fenster) < 2:
        return None
    hoch = float(fenster.iloc[0])
    if hoch <= 0:
        return None
    drawdowns = (hoch - fenster) / hoch
    return float(drawdowns.max())


def _fenster_am_hoch_verankert(preise: pd.Series) -> pd.Series:
    """Gibt Teilserie zurück, die am Maximum (52W-Hoch) beginnt."""
    if preise.empty:
        return preise
    hoch_idx = preise.idxmax()
    return preise.loc[hoch_idx:]


def _tage_seit_hoch(preise: pd.Series) -> Optional[int]:
    """Anzahl Handelstage vom 52W-Hoch bis heute."""
    if preise.empty:
        return None
    hoch_idx = preise.idxmax()
    fenster = preise.loc[hoch_idx:]
    return len(fenster) - 1  # -1: Hoch-Tag selbst nicht mitzählen


def _befund_drawdown_aktuell(dd: Optional[float]) -> Befund:
    label = "Drawdown vom 52W-Hoch"
    if dd is None:
        return befund_unbestimmt(label)
    if dd > SCHWELLEN["drawdown_extrem_max"]:
        return Befund(
            label=label, wert=round(dd, 4), bestanden=False,
            zustand=Zustand.NICHT_PASSIERT,
            details=f"Extremer Drawdown > {SCHWELLEN['drawdown_extrem_max']:.0%} – Value Trap prüfen",
        )
    bestanden = dd >= SCHWELLEN["drawdown_guenstig_min"]
    return Befund(
        label=label, wert=round(dd, 4), bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=(
            f"Günstig: ≥ {SCHWELLEN['drawdown_guenstig_min']:.0%} unter 52W-Hoch"
            if bestanden else
            f"Noch zu nah am 52W-Hoch (Schwelle: ≥ {SCHWELLEN['drawdown_guenstig_min']:.0%})"
        ),
    )


def _befund_naehe_tief(max_dd: Optional[float], akt_dd: Optional[float]) -> Befund:
    """Nähe zum Tief seit 52W-Hoch: Drawdown vs. Drawdown (kein Rendite-Vergleich)."""
    label = "Nähe zum Tief im 52W-Fenster (DD vs. DD)"
    if max_dd is None or akt_dd is None:
        return befund_unbestimmt(label)
    if max_dd <= 0:
        return befund_unbestimmt(label, "Max-Drawdown im Fenster ist 0 – Aktie am Allzeithoch")
    ratio = akt_dd / max_dd
    bestanden = ratio >= SCHWELLEN["naehe_tief_faktor"]
    return Befund(
        label=label, wert=round(ratio, 4), bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=(
            f"Akt. DD ist {ratio:.0%} des Max-DD im Fenster "
            f"(Schwelle ≥ {SCHWELLEN['naehe_tief_faktor']:.0%})"
        ),
    )


def _befund_hist_vergleich(hist_dd: Optional[float], akt_dd: Optional[float]) -> Befund:
    """Vergleich: Akt. Drawdown vs. Vorjahres-Drawdown. Drawdown gegen Drawdown."""
    label = "Drawdown vs. Vorjahres-Drawdown"
    if hist_dd is None:
        return befund_unbestimmt(label, "Nicht genug Vorjahresdaten")
    if akt_dd is None:
        return befund_unbestimmt(label)
    if hist_dd <= 0:
        return befund_unbestimmt(label, "Vorjahr: kein Drawdown messbar")
    ratio = akt_dd / hist_dd
    bestanden = ratio >= SCHWELLEN["hist_vergleich_faktor"]
    return Befund(
        label=label, wert=round(akt_dd, 4), bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=(
            f"Akt. DD {akt_dd:.1%} vs. Vorjahres-DD {hist_dd:.1%} "
            f"(Ratio {ratio:.0%}, Schwelle ≥ {SCHWELLEN['hist_vergleich_faktor']:.0%})"
        ),
    )


def _befund_dauer(tage: Optional[int]) -> Befund:
    """Drawdown-Dauer: etablierte Schwäche, kein flüchtiger Dip, kein Value Trap."""
    label = "Drawdown-Dauer (Handelstage seit 52W-Hoch)"
    if tage is None:
        return befund_unbestimmt(label)
    if tage < SCHWELLEN["dauer_min_tage"]:
        return Befund(
            label=label, wert=tage, bestanden=False,
            zustand=Zustand.NICHT_PASSIERT,
            details=(
                f"Zu frisch: {tage} Tage (< {SCHWELLEN['dauer_min_tage']}) – "
                "könnte kurzfristiger Dip sein"
            ),
        )
    if tage > SCHWELLEN["dauer_max_tage"]:
        return Befund(
            label=label, wert=tage, bestanden=False,
            zustand=Zustand.NICHT_PASSIERT,
            details=(
                f"Sehr lang: {tage} Tage (> {SCHWELLEN['dauer_max_tage']}) – "
                "Value Trap prüfen"
            ),
        )
    return Befund(
        label=label, wert=tage, bestanden=True,
        zustand=Zustand.PASSIERT,
        details=f"Etabliert: {tage} Handelstage seit 52W-Hoch",
    )


def analyse(ticker: str) -> List[Befund]:
    """Drawdown-Analyse mit am 52W-Hoch verankertem Fenster. 4 Befunde."""
    try:
        daten = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    except Exception as exc:
        return [befund_unbestimmt(f"Drawdown {ticker}", f"Datenabruf: {exc}")]

    if daten.empty or "Close" not in daten.columns:
        return [befund_unbestimmt(f"Drawdown {ticker}", "Keine Kursdaten")]

    close = daten["Close"].dropna()
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    close = close.squeeze()

    if len(close) < 20:
        return [befund_unbestimmt(f"Drawdown {ticker}", "Zu wenig Datenpunkte")]

    n = SCHWELLEN["fenster_tage"]

    # Aktuelles 52W-Fenster
    letzte_52w = close.tail(n)
    akt_dd = _drawdown_aktuell(letzte_52w)

    # Fenster ab 52W-Hoch → Drawdown vs. Drawdown (kein Rendite-Bias)
    fenster_seit_hoch = _fenster_am_hoch_verankert(letzte_52w)
    max_dd = _max_dd_im_fenster(fenster_seit_hoch)

    # Dauer seit 52W-Hoch
    tage = _tage_seit_hoch(letzte_52w)

    # Vorjahr für historischen Vergleich
    prior = close.iloc[:-n] if len(close) > n else pd.Series(dtype=float)
    hist_dd: Optional[float] = None
    if len(prior) >= 20:
        prior_fenster = _fenster_am_hoch_verankert(prior)
        hist_dd = _max_dd_im_fenster(prior_fenster)

    return [
        _befund_drawdown_aktuell(akt_dd),
        _befund_naehe_tief(max_dd, akt_dd),
        _befund_hist_vergleich(hist_dd, akt_dd),
        _befund_dauer(tage),
    ]


def conviction(befunde: List[Befund]) -> float:
    """Score 0–100: Anteil bestandener, bewertbarer Kriterien × 100."""
    bewertbar = [b for b in befunde if b.bestanden is not None]
    if not bewertbar:
        return 0.0
    return round(sum(1 for b in bewertbar if b.bestanden) / len(bewertbar) * 100, 1)

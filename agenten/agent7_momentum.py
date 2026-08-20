"""
Agent 7 – Momentum.

Beantwortet: Befindet sich die Aktie in einer günstigen Momentum-Phase?

1. 12M-Momentum            – Preisrückgang ≥ -15% (kein Momentum-Crash)
2. 6M-Relative Stärke      – 6M-Rendite vs. SPY ≥ -15%
3. RSI-Zone                – 14-Tage RSI zwischen 30 und 65 (Value-Zone)
4. 200MA-Abstand           – Kurs ≤ 50% über 200-Tage-Durchschnitt (nicht überkauft)
"""
from typing import List, Optional

import pandas as pd
import yfinance as yf

from kern.typen import Befund, Zustand, befund_unbestimmt

SCHWELLEN = {
    "momentum_12m_min": -0.15,        # 12M-Rendite ≥ -15%
    "relative_staerke_min": -0.15,    # 6M-Rendite vs. SPY ≥ -15%
    "rsi_min": 30.0,                  # RSI-Untergrenze (extreme Panik)
    "rsi_max": 65.0,                  # RSI-Obergrenze (überkauft)
    "ma200_aufschlag_max": 0.50,      # Kurs max. 50% über 200MA
    "conviction_gewicht": 0.125,      # 8 Agenten → gleiches Gewicht
}

_PERIODEN_12M = 252
_PERIODEN_6M = 126
_PERIODEN_RSI = 14


# ── Technische Hilfs­funktionen ───────────────────────────────────────────────

def _rsi(preise: pd.Series, perioden: int = _PERIODEN_RSI) -> Optional[float]:
    """Exponentiell geglätteter RSI (Wilder-Methode über EWM)."""
    if len(preise) < perioden + 1:
        return None

    delta = preise.diff().dropna()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=perioden - 1, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(com=perioden - 1, adjust=False).mean().iloc[-1]

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 2)


# ── Befund-Funktionen ─────────────────────────────────────────────────────────

def _momentum_12m(preise: pd.Series) -> Befund:
    """12M-Kursmomentum: Jahresrendite ≥ Schwelle → kein anhaltender Absturz."""
    label = "12M-Kursmomentum"

    if len(preise) < _PERIODEN_12M:
        return befund_unbestimmt(label, "Nicht genug Kursdaten (mind. 252 Handelstage nötig)")

    aktuell = float(preise.iloc[-1])
    vor_12m = float(preise.iloc[-_PERIODEN_12M])

    if vor_12m <= 0:
        return befund_unbestimmt(label, "Basiswert vor 12M nicht positiv")

    rendite = (aktuell - vor_12m) / vor_12m
    bestanden = rendite >= SCHWELLEN["momentum_12m_min"]
    return Befund(
        label=label,
        wert=round(rendite, 4),
        bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=(
            f"{rendite:+.1%} in 12M "
            f"(Schwelle ≥ {SCHWELLEN['momentum_12m_min']:.0%})"
        ),
    )


def _relative_staerke_6m(ticker_preise: pd.Series, spy_preise: pd.Series) -> Befund:
    """6M-Relative Stärke: Ticker-Rendite minus SPY-Rendite."""
    label = "6M-Relative Stärke vs. SPY"

    if ticker_preise is None or len(ticker_preise) < _PERIODEN_6M:
        return befund_unbestimmt(label, "Nicht genug Ticker-Kursdaten (mind. 126 Handelstage)")

    if spy_preise is None or len(spy_preise) < _PERIODEN_6M:
        return befund_unbestimmt(label, "Nicht genug SPY-Kursdaten")

    def _rendite(preise: pd.Series) -> Optional[float]:
        if len(preise) < _PERIODEN_6M:
            return None
        akt = float(preise.iloc[-1])
        alt = float(preise.iloc[-_PERIODEN_6M])
        return (akt - alt) / alt if alt > 0 else None

    r_ticker = _rendite(ticker_preise)
    r_spy = _rendite(spy_preise)

    if r_ticker is None or r_spy is None:
        return befund_unbestimmt(label, "Basiswert nicht positiv")

    delta = r_ticker - r_spy
    bestanden = delta >= SCHWELLEN["relative_staerke_min"]
    return Befund(
        label=label,
        wert=round(delta, 4),
        bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=(
            f"Ticker {r_ticker:+.1%} vs. SPY {r_spy:+.1%} → Delta {delta:+.1%} "
            f"(Schwelle ≥ {SCHWELLEN['relative_staerke_min']:.0%})"
        ),
    )


def _rsi_zone(rsi_wert: Optional[float]) -> Befund:
    """RSI-Zone: 30–65 = Value-Zone (weder Panik noch überkauft)."""
    label = f"RSI-Zone ({_PERIODEN_RSI}-Tage)"

    if rsi_wert is None:
        return befund_unbestimmt(label, "RSI nicht berechenbar (zu wenig Daten)")

    in_zone = SCHWELLEN["rsi_min"] <= rsi_wert <= SCHWELLEN["rsi_max"]
    if in_zone:
        return Befund(
            label=label,
            wert=round(rsi_wert, 2),
            bestanden=True,
            zustand=Zustand.PASSIERT,
            details=(
                f"RSI {rsi_wert:.1f} in Value-Zone "
                f"[{SCHWELLEN['rsi_min']:.0f}–{SCHWELLEN['rsi_max']:.0f}]"
            ),
        )

    if rsi_wert > SCHWELLEN["rsi_max"]:
        details = f"RSI {rsi_wert:.1f} – überkauft (> {SCHWELLEN['rsi_max']:.0f})"
    else:
        details = f"RSI {rsi_wert:.1f} – extremer Ausverkauf (< {SCHWELLEN['rsi_min']:.0f})"

    return Befund(
        label=label,
        wert=round(rsi_wert, 2),
        bestanden=False,
        zustand=Zustand.NICHT_PASSIERT,
        details=details,
    )


def _ma200_abstand(preise: pd.Series) -> Befund:
    """Kurs vs. 200-Tage-Durchschnitt: nicht mehr als 50% darüber (overbought-Schutz)."""
    label = "Abstand zum 200MA"

    if len(preise) < 200:
        return befund_unbestimmt(label, "Nicht genug Kursdaten für 200MA (mind. 200 Handelstage)")

    ma200 = float(preise.tail(200).mean())
    aktuell = float(preise.iloc[-1])

    if ma200 <= 0:
        return befund_unbestimmt(label, "200MA nicht positiv")

    aufschlag = (aktuell - ma200) / ma200
    bestanden = aufschlag <= SCHWELLEN["ma200_aufschlag_max"]
    return Befund(
        label=label,
        wert=round(aufschlag, 4),
        bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=(
            f"Kurs {aktuell:.2f} ist {aufschlag:+.1%} gegenüber 200MA {ma200:.2f} "
            f"(Schwelle ≤ +{SCHWELLEN['ma200_aufschlag_max']:.0%})"
        ),
    )


# ── Haupt-API ─────────────────────────────────────────────────────────────────

def analyse(ticker: str) -> List[Befund]:
    """Momentum-Analyse. Gibt immer 4 Befund-Objekte zurück."""
    try:
        daten = yf.download(ticker, period="15mo", progress=False, auto_adjust=True)
        spy_daten = yf.download("SPY", period="7mo", progress=False, auto_adjust=True)
    except Exception as exc:
        lbl = f"Momentum {ticker}"
        return [befund_unbestimmt(lbl, f"Datenabruf: {exc}") for _ in range(4)]

    def _close_serie(df: pd.DataFrame) -> pd.Series:
        if df is None or df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float)
        s = df["Close"].squeeze()
        if hasattr(s, "columns"):
            s = s.iloc[:, 0]
        return s.dropna()

    close = _close_serie(daten)
    spy_close = _close_serie(spy_daten)

    rsi_wert = _rsi(close) if len(close) >= _PERIODEN_RSI + 1 else None

    return [
        _momentum_12m(close),
        _relative_staerke_6m(close, spy_close),
        _rsi_zone(rsi_wert),
        _ma200_abstand(close),
    ]


def conviction(befunde: List[Befund]) -> float:
    """Score 0–100: Anteil bestandener, bewertbarer Kriterien × 100."""
    bewertbar = [b for b in befunde if b.bestanden is not None]
    if not bewertbar:
        return 0.0
    return round(sum(1 for b in bewertbar if b.bestanden) / len(bewertbar) * 100, 1)

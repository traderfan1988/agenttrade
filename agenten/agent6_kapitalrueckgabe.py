"""
Agent 6 – Kapitalrückgabe.

Beantwortet: Gibt das Unternehmen Kapital effizient an Aktionäre zurück?

1. Dividendenrendite     – ≥ 2% (UNBESTIMMT wenn kein Dividendenzahler)
2. FCF-Ausschüttungsquote – Dividende / FCF ≤ 80% (nur wenn Dividende vorhanden)
3. Buyback-Rendite        – Netto-Rückkäufe / MarketCap ≥ 1.5%
4. Shareholder Yield      – Dividende + Buybacks gesamt ≥ 3% des MarketCap
"""
from typing import List, Optional

import pandas as pd
import yfinance as yf

from kern.typen import Befund, Zustand, befund_unbestimmt

SCHWELLEN = {
    "dividenden_rendite_min": 0.02,   # ≥ 2% Dividendenrendite
    "fcf_payout_max": 0.80,           # FCF-Ausschüttungsquote ≤ 80%
    "buyback_yield_min": 0.015,       # Buyback-Rendite ≥ 1.5%
    "shareholder_yield_min": 0.03,    # Gesamt-Kapitalrückgabe ≥ 3%
    "conviction_gewicht": 0.125,      # 8 Agenten → gleiches Gewicht
}


# ── Datenhilfs­funktionen ─────────────────────────────────────────────────────

def _buyback_aus_cashflow(cf) -> Optional[float]:
    """Aktienrückkäufe (absolut) aus Cashflow-Statement. yfinance: Wert ist negativ."""
    if cf is None or (hasattr(cf, "empty") and cf.empty):
        return None
    for feld in ("Repurchase Of Capital Stock", "Common Stock Repurchased",
                 "Purchase Of Business"):
        try:
            serie = cf.loc[feld].dropna()
            if not serie.empty:
                val = float(serie.iloc[0])
                if val < 0:
                    return abs(val)
        except KeyError:
            continue
    return None


def _dividenden_aus_cashflow(cf) -> Optional[float]:
    """Gezahlte Dividenden (absolut) aus Cashflow. yfinance: Wert ist negativ."""
    if cf is None or (hasattr(cf, "empty") and cf.empty):
        return None
    for feld in ("Common Stock Dividends Paid", "Cash Dividends Paid",
                 "Dividends Paid"):
        try:
            serie = cf.loc[feld].dropna()
            if not serie.empty:
                val = float(serie.iloc[0])
                return abs(val)
        except KeyError:
            continue
    return None


# ── Befund-Funktionen ─────────────────────────────────────────────────────────

def _dividendenrendite(yield_pct: Optional[float]) -> Befund:
    """Dividendenrendite. 0 / None → UNBESTIMMT (Nicht-Zahler werden nicht bestraft)."""
    label = "Dividendenrendite"

    if yield_pct is None or yield_pct == 0.0:
        return befund_unbestimmt(label, "Kein Dividendenzahler – Kapitalrückgabe via Buyback prüfen")

    bestanden = yield_pct >= SCHWELLEN["dividenden_rendite_min"]
    return Befund(
        label=label,
        wert=round(yield_pct, 4),
        bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=f"{yield_pct:.2%} Rendite (Schwelle ≥ {SCHWELLEN['dividenden_rendite_min']:.0%})",
    )


def _fcf_payout(
    fcf: Optional[float],
    net_income: Optional[float],  # noqa: ARG001
    dividenden_gezahlt: Optional[float],
) -> Befund:
    """FCF-Ausschüttungsquote: Dividende / FCF. Nur bewertbar wenn Dividende vorhanden."""
    label = "FCF-Ausschüttungsquote"

    if dividenden_gezahlt is None or dividenden_gezahlt == 0:
        return befund_unbestimmt(label, "Kein Dividendenzahler – nicht anwendbar")

    if fcf is None or fcf <= 0:
        return befund_unbestimmt(label, "FCF nicht verfügbar oder negativ")

    ratio = dividenden_gezahlt / fcf
    bestanden = ratio <= SCHWELLEN["fcf_payout_max"]
    return Befund(
        label=label,
        wert=round(ratio, 4),
        bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=f"{ratio:.0%} des FCF ausgeschüttet (Schwelle ≤ {SCHWELLEN['fcf_payout_max']:.0%})",
    )


def _buyback_yield(
    buyback: Optional[float],
    market_cap: Optional[float],
) -> Befund:
    """Aktienrückkauf-Rendite: Rückkäufe / MarketCap."""
    label = "Buyback-Rendite"

    if market_cap is None or market_cap <= 0:
        return befund_unbestimmt(label, "Marktkapitalisierung nicht verfügbar")

    if buyback is None:
        return befund_unbestimmt(label, "Rückkauf-Daten nicht verfügbar")

    yield_val = buyback / market_cap
    bestanden = yield_val >= SCHWELLEN["buyback_yield_min"]
    return Befund(
        label=label,
        wert=round(yield_val, 4),
        bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=(
            f"{yield_val:.2%} Buyback-Rendite "
            f"(Schwelle ≥ {SCHWELLEN['buyback_yield_min']:.1%})"
        ),
    )


def _shareholder_yield(
    dividenden: Optional[float],
    buyback: Optional[float],
    market_cap: Optional[float],
) -> Befund:
    """Gesamt-Kapitalrückgabe: (Dividende + Buybacks) / MarketCap."""
    label = "Shareholder Yield (Dividende + Buybacks)"

    if market_cap is None or market_cap <= 0:
        return befund_unbestimmt(label, "Marktkapitalisierung nicht verfügbar")

    if dividenden is None and buyback is None:
        return befund_unbestimmt(label, "Keine Kapitalrückgabe-Daten verfügbar")

    gesamt = (dividenden or 0.0) + (buyback or 0.0)
    yield_val = gesamt / market_cap
    bestanden = yield_val >= SCHWELLEN["shareholder_yield_min"]
    return Befund(
        label=label,
        wert=round(yield_val, 4),
        bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=(
            f"{yield_val:.2%} gesamt ({(dividenden or 0)/market_cap:.2%} Dividende + "
            f"{(buyback or 0)/market_cap:.2%} Buyback, Schwelle ≥ {SCHWELLEN['shareholder_yield_min']:.0%})"
        ),
    )


# ── Haupt-API ─────────────────────────────────────────────────────────────────

def analyse(ticker: str) -> List[Befund]:
    """Kapitalrückgabe-Analyse. Gibt immer 4 Befund-Objekte zurück."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        cashflow = t.cashflow
        financials = t.financials
    except Exception as exc:
        lbl = f"Kapitalrückgabe {ticker}"
        return [befund_unbestimmt(lbl, f"Datenabruf: {exc}") for _ in range(4)]

    market_cap = info.get("marketCap")
    fcf = info.get("freeCashflow")

    # yfinance 1.2.0: dividendYield kommt als Dezimalbruch (0.0075 = 0.75%).
    # Heuristik: Werte > 0.5 sind bereits Prozent (z.B. 0.75 = 0.75%) → durch 100 teilen.
    div_yield = info.get("dividendYield")
    if div_yield is not None and div_yield > 0.5:
        div_yield = div_yield / 100

    buyback = _buyback_aus_cashflow(cashflow)
    dividenden_gezahlt = _dividenden_aus_cashflow(cashflow)

    # FCF aus cashflow bevorzugen
    if fcf is None and cashflow is not None and not (hasattr(cashflow, "empty") and cashflow.empty):
        try:
            serie = cashflow.loc["Free Cash Flow"].dropna()
            if not serie.empty:
                fcf = float(serie.iloc[0])
        except KeyError:
            pass

    # Net Income für FCF-Payout
    net_income = None
    if financials is not None and not (hasattr(financials, "empty") and financials.empty):
        try:
            ni_serie = financials.loc["Net Income"].dropna()
            if not ni_serie.empty:
                net_income = float(ni_serie.iloc[0])
        except KeyError:
            pass

    return [
        _dividendenrendite(div_yield),
        _fcf_payout(fcf, net_income, dividenden_gezahlt),
        _buyback_yield(buyback, market_cap),
        _shareholder_yield(dividenden_gezahlt, buyback, market_cap),
    ]


def conviction(befunde: List[Befund]) -> float:
    """Score 0–100: Anteil bestandener, bewertbarer Kriterien × 100."""
    bewertbar = [b for b in befunde if b.bestanden is not None]
    if not bewertbar:
        return 0.0
    return round(sum(1 for b in bewertbar if b.bestanden) / len(bewertbar) * 100, 1)

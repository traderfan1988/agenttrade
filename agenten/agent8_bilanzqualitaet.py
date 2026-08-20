"""
Agent 8 – Bilanzqualität.

Beantwortet: Ist die Bilanz solide und tragfähig?

1. Net Debt / EBITDA   – ≤ 3× (EBITDA-basierte Verschuldungsmetrik)
2. Current Ratio       – ≥ 1.5 (kurzfristige Liquidität)
3. Goodwill-Anteil     – ≤ 30% der Bilanzsumme (Akquisitions-Risiko)
4. Working Capital     – positiv (basale Liquiditätssicherheit)
"""
from typing import List, Optional

import pandas as pd
import yfinance as yf

from kern.typen import Befund, Zustand, befund_unbestimmt

SCHWELLEN = {
    "net_debt_ebitda_max": 3.0,       # Net Debt/EBITDA ≤ 3×
    "current_ratio_min": 1.5,         # Current Ratio ≥ 1.5
    "goodwill_anteil_max": 0.30,      # Goodwill ≤ 30% Total Assets
    "conviction_gewicht": 0.125,      # 8 Agenten → gleiches Gewicht
}


# ── Datenhilfs­funktionen ─────────────────────────────────────────────────────

def _ebitda_aus_financials(fin) -> Optional[float]:
    """EBITDA aus Finanzdaten."""
    if fin is None or (hasattr(fin, "empty") and fin.empty):
        return None
    for feld in ("EBITDA", "Normalized EBITDA"):
        try:
            serie = fin.loc[feld].dropna()
            if not serie.empty:
                return float(serie.iloc[0])
        except KeyError:
            continue
    return None


def _bs_wert(bs, *felder) -> Optional[float]:
    """Ersten verfügbaren Wert aus der Balance-Sheet-Zeile lesen."""
    if bs is None or (hasattr(bs, "empty") and bs.empty):
        return None
    for feld in felder:
        try:
            serie = bs.loc[feld].dropna()
            if not serie.empty:
                return float(serie.iloc[0])
        except KeyError:
            continue
    return None


# ── Befund-Funktionen ─────────────────────────────────────────────────────────

def _net_debt_ebitda(
    total_debt: Optional[float],
    cash: Optional[float],
    ebitda: Optional[float],
) -> Befund:
    """Net Debt / EBITDA. Nettoliquid (Net Debt ≤ 0) → PASSIERT."""
    label = "Net Debt / EBITDA"

    if total_debt is None and cash is None:
        return befund_unbestimmt(label, "Verschuldungsdaten nicht verfügbar")

    if ebitda is None:
        return befund_unbestimmt(label, "EBITDA nicht verfügbar")

    if ebitda <= 0:
        return befund_unbestimmt(label, "EBITDA nicht positiv – Kennzahl nicht aussagekräftig")

    net_debt = (total_debt or 0.0) - (cash or 0.0)

    if net_debt <= 0:
        return Befund(
            label=label,
            wert=round(net_debt / ebitda, 2),
            bestanden=True,
            zustand=Zustand.PASSIERT,
            details="Nettoliquid: Cash übersteigt Schulden",
        )

    ratio = net_debt / ebitda
    bestanden = ratio <= SCHWELLEN["net_debt_ebitda_max"]
    return Befund(
        label=label,
        wert=round(ratio, 2),
        bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=f"{ratio:.1f}× (Schwelle ≤ {SCHWELLEN['net_debt_ebitda_max']:.0f}×)",
    )


def _current_ratio(
    current_assets: Optional[float],
    current_liabilities: Optional[float],
) -> Befund:
    """Current Ratio: kurzfristige Vermögenswerte / kurzfristige Verbindlichkeiten."""
    label = "Current Ratio (Kurzfrist-Liquidität)"

    if current_assets is None or current_liabilities is None:
        return befund_unbestimmt(label, "Bilanzdaten für Liquiditätsberechnung fehlen")

    if current_liabilities <= 0:
        return befund_unbestimmt(label, "Kurzfristige Verbindlichkeiten ≤ 0")

    ratio = current_assets / current_liabilities
    bestanden = ratio >= SCHWELLEN["current_ratio_min"]
    return Befund(
        label=label,
        wert=round(ratio, 2),
        bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=f"{ratio:.2f}× (Schwelle ≥ {SCHWELLEN['current_ratio_min']:.1f}×)",
    )


def _goodwill_anteil(
    goodwill: Optional[float],
    total_assets: Optional[float],
) -> Befund:
    """Goodwill als Anteil der Bilanzsumme: Akquisitions-Risiko."""
    label = "Goodwill / Bilanzsumme"

    if total_assets is None or total_assets <= 0:
        return befund_unbestimmt(label, "Bilanzsumme nicht verfügbar")

    if goodwill is None:
        return befund_unbestimmt(label, "Goodwill-Daten nicht verfügbar")

    anteil = goodwill / total_assets
    bestanden = anteil <= SCHWELLEN["goodwill_anteil_max"]
    return Befund(
        label=label,
        wert=round(anteil, 4),
        bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=(
            f"{anteil:.1%} der Bilanzsumme ist Goodwill "
            f"(Schwelle ≤ {SCHWELLEN['goodwill_anteil_max']:.0%})"
        ),
    )


def _working_capital(
    current_assets: Optional[float],
    current_liabilities: Optional[float],
) -> Befund:
    """Working Capital (Current Assets − Current Liabilities) muss positiv sein."""
    label = "Working Capital"

    if current_assets is None or current_liabilities is None:
        return befund_unbestimmt(label, "Bilanzdaten für Working Capital fehlen")

    wc = current_assets - current_liabilities
    bestanden = wc > 0
    return Befund(
        label=label,
        wert=round(wc),
        bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=(
            f"Working Capital: {wc:+,.0f} "
            f"({'positiv' if bestanden else 'negativ – kurzfristig illiquid'})"
        ),
    )


# ── Haupt-API ─────────────────────────────────────────────────────────────────

def analyse(ticker: str) -> List[Befund]:
    """Bilanzqualitäts-Analyse. Gibt immer 4 Befund-Objekte zurück."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        bs = t.balance_sheet
        financials = t.financials
    except Exception as exc:
        lbl = f"Bilanzqualität {ticker}"
        return [befund_unbestimmt(lbl, f"Datenabruf: {exc}") for _ in range(4)]

    total_debt = info.get("totalDebt")
    cash = info.get("totalCash")
    ebitda = _ebitda_aus_financials(financials) or info.get("ebitda")

    current_assets = _bs_wert(bs, "Current Assets", "Total Current Assets")
    current_liabilities = _bs_wert(bs, "Current Liabilities", "Total Current Liabilities")
    goodwill = _bs_wert(bs, "Goodwill", "Goodwill And Other Intangible Assets")
    total_assets = _bs_wert(bs, "Total Assets")

    return [
        _net_debt_ebitda(total_debt, cash, ebitda),
        _current_ratio(current_assets, current_liabilities),
        _goodwill_anteil(goodwill, total_assets),
        _working_capital(current_assets, current_liabilities),
    ]


def conviction(befunde: List[Befund]) -> float:
    """Score 0–100: Anteil bestandener, bewertbarer Kriterien × 100."""
    bewertbar = [b for b in befunde if b.bestanden is not None]
    if not bewertbar:
        return 0.0
    return round(sum(1 for b in bewertbar if b.bestanden) / len(bewertbar) * 100, 1)

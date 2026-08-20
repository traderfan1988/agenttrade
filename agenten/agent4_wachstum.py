"""
Agent 4 – Wachstumsanalyse.

Beantwortet: Wächst das Unternehmen oder ist es eine Value Trap?

Günstiger Preis + schrumpfendes Geschäft = Value Trap.
Günstiger Preis + wachsendes Geschäft = Value Opportunity.

4 Befunde:
  1. Umsatz-CAGR 3J  – organisches Wachstum der Topline (≥ 5%)
  2. Gewinn-CAGR 3J  – Profitabilitätsentwicklung (≥ 5%)
  3. FCF-CAGR 3J     – Wachstum des echten Cashflows (≥ 3%)
  4. Operating Leverage – Op.Income wächst schneller als Umsatz (Skalierbarkeit)
"""
from typing import List, Optional

import pandas as pd
import yfinance as yf

from kern.typen import Befund, Zustand, befund_unbestimmt

SCHWELLEN = {
    "umsatz_cagr_min": 0.05,        # Revenue CAGR 3J ≥ 5%
    "gewinn_cagr_min": 0.05,        # Net Income CAGR 3J ≥ 5%
    "fcf_cagr_min": 0.03,           # FCF CAGR 3J ≥ 3%
    "mindest_jahre": 3,             # mind. 3 Wachstumsjahre = 4 Datenpunkte nötig
    "conviction_gewicht": 0.125,    # 8 Agenten → gleiches Gewicht
}


def _cagr(serie: pd.Series, n_jahre: int = 3) -> Optional[float]:
    """Compound Annual Growth Rate über n_jahre.

    Serie hat neueste Werte zuerst (yfinance-Format).
    Gibt None zurück wenn Datenbasis zu dünn oder Basis-/Endwert ≤ 0.
    """
    clean = serie.dropna()
    if len(clean) <= n_jahre:
        return None
    newest = float(clean.iloc[0])
    oldest = float(clean.iloc[n_jahre])
    if oldest <= 0 or newest <= 0:
        return None
    return (newest / oldest) ** (1.0 / n_jahre) - 1


def _umsatz_cagr(fin) -> Befund:
    """Umsatz-CAGR über 3 Jahre. Prüft organisches Topline-Wachstum."""
    label = "Umsatz-CAGR (3J)"
    if fin is None or (hasattr(fin, "empty") and fin.empty):
        return befund_unbestimmt(label, "Finanzdaten fehlen")
    try:
        revenue = fin.loc["Total Revenue"].dropna()
    except KeyError:
        return befund_unbestimmt(label, "Umsatzdaten nicht verfügbar")

    n = SCHWELLEN["mindest_jahre"]
    cagr = _cagr(revenue, n)
    if cagr is None:
        return befund_unbestimmt(
            label,
            f"Zu wenig Daten oder Basiswert ≤ 0 (mind. {n + 1} Jahre nötig)",
        )
    bestanden = cagr >= SCHWELLEN["umsatz_cagr_min"]
    return Befund(
        label=label, wert=round(cagr, 4), bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=f"CAGR {cagr:.1%} (Schwelle ≥ {SCHWELLEN['umsatz_cagr_min']:.0%})",
    )


def _gewinn_cagr(fin) -> Befund:
    """Net-Income-CAGR über 3 Jahre. Prüft Profitabilitätsentwicklung."""
    label = "Gewinn-CAGR (3J)"
    if fin is None or (hasattr(fin, "empty") and fin.empty):
        return befund_unbestimmt(label, "Finanzdaten fehlen")
    try:
        net_income = fin.loc["Net Income"].dropna()
    except KeyError:
        return befund_unbestimmt(label, "Gewinndaten nicht verfügbar")

    n = SCHWELLEN["mindest_jahre"]
    cagr = _cagr(net_income, n)
    if cagr is None:
        return befund_unbestimmt(
            label,
            f"Zu wenig Daten oder Verlust im Basisjahr (mind. {n + 1} Jahre nötig)",
        )
    bestanden = cagr >= SCHWELLEN["gewinn_cagr_min"]
    return Befund(
        label=label, wert=round(cagr, 4), bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=f"CAGR {cagr:.1%} (Schwelle ≥ {SCHWELLEN['gewinn_cagr_min']:.0%})",
    )


def _fcf_cagr(cashflow) -> Befund:
    """FCF-CAGR über 3 Jahre. Prüft Wachstum des echten Cashflows."""
    label = "FCF-CAGR (3J)"
    if cashflow is None or (hasattr(cashflow, "empty") and cashflow.empty):
        return befund_unbestimmt(label, "Cashflow-Daten fehlen")
    try:
        fcf = cashflow.loc["Free Cash Flow"].dropna()
    except KeyError:
        return befund_unbestimmt(label, "Free Cash Flow nicht verfügbar")

    n = SCHWELLEN["mindest_jahre"]
    cagr = _cagr(fcf, n)
    if cagr is None:
        return befund_unbestimmt(
            label,
            f"Zu wenig Daten oder negativer FCF im Basisjahr (mind. {n + 1} Jahre nötig)",
        )
    bestanden = cagr >= SCHWELLEN["fcf_cagr_min"]
    return Befund(
        label=label, wert=round(cagr, 4), bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=f"CAGR {cagr:.1%} (Schwelle ≥ {SCHWELLEN['fcf_cagr_min']:.0%})",
    )


def _operating_leverage(fin) -> Befund:
    """Operating Leverage: Operating Income wächst schneller als Umsatz → skalierbar."""
    label = "Operating Leverage (Skalierbarkeit)"
    if fin is None or (hasattr(fin, "empty") and fin.empty):
        return befund_unbestimmt(label, "Finanzdaten fehlen")
    try:
        revenue = fin.loc["Total Revenue"].dropna()
    except KeyError:
        return befund_unbestimmt(label, "Umsatzdaten nicht verfügbar")

    op_income = None
    for feld in ("Operating Income", "EBIT"):
        try:
            serie = fin.loc[feld].dropna()
            if not serie.empty:
                op_income = serie
                break
        except KeyError:
            continue
    if op_income is None:
        return befund_unbestimmt(label, "Operating Income nicht verfügbar")

    n = 2  # 2-Jahres-CAGR für Leverage-Check (stabiler als 3J)
    rev_cagr = _cagr(revenue, n)
    op_cagr = _cagr(op_income, n)

    if rev_cagr is None or op_cagr is None:
        return befund_unbestimmt(label, "Zu wenig Daten für CAGR-Vergleich")

    bestanden = op_cagr > rev_cagr
    delta = op_cagr - rev_cagr
    return Befund(
        label=label, wert=round(delta, 4), bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=(
            f"Op.Income CAGR {op_cagr:.1%} > Umsatz CAGR {rev_cagr:.1%} – skalierbar"
            if bestanden else
            f"Op.Income CAGR {op_cagr:.1%} ≤ Umsatz CAGR {rev_cagr:.1%} – keine Skalierung"
        ),
    )


def analyse(ticker: str) -> List[Befund]:
    """Wachstumsanalyse. Gibt immer 4 Befund-Objekte zurück."""
    try:
        t = yf.Ticker(ticker)
        financials = t.financials
        cashflow = t.cashflow
    except Exception as exc:
        return [befund_unbestimmt(f"Wachstum {ticker}", f"Datenabruf: {exc}")]

    if financials is None or (hasattr(financials, "empty") and financials.empty):
        financials = pd.DataFrame()
    if cashflow is None or (hasattr(cashflow, "empty") and cashflow.empty):
        cashflow = pd.DataFrame()

    return [
        _umsatz_cagr(financials),
        _gewinn_cagr(financials),
        _fcf_cagr(cashflow),
        _operating_leverage(financials),
    ]


def conviction(befunde: List[Befund]) -> float:
    """Score 0–100: Anteil bestandener, bewertbarer Kriterien × 100."""
    bewertbar = [b for b in befunde if b.bestanden is not None]
    if not bewertbar:
        return 0.0
    return round(sum(1 for b in bewertbar if b.bestanden) / len(bewertbar) * 100, 1)

"""
Agent 1 – Fundamentale Bewertungs- und Qualitätsanalyse.

Phase 2: ROCE, Zinsdeckung (mit Nettoliquiditäts-Handling), Cash Conversion Rate.
Phase 3: EV/EBIT als Primär-Bewertungsmetrik, FCF/EV-Rendite statt KGV/FCF-MarketCap.

7 Befunde:
  1. EV/EBIT            – kapitalstrukturneutrale Bewertung
  2. FCF/EV-Rendite     – echter Cash auf Gesamtunternehmensbasis
  3. KBV (P/B)          – Substanzbewertung
  4. Cash Conversion Rate (FCF/NetIncome) – Gewinnqualität
  5. ROCE               – Kapitalrendite (Finanz-/Utility-/REIT-Sektoren ausgeschlossen)
  6. Zinsdeckungsgrad   – Schuldenservice; Nettoliquid → immer PASSIERT
  7. Schulden/EK        – Bilanzstabilität
"""
from typing import List, Optional

import pandas as pd
import yfinance as yf

from kern.typen import Befund, Zustand, befund_unbestimmt

SCHWELLEN = {
    "ev_ebit_max": 15.0,            # EV/EBIT Obergrenze
    "fcf_ev_yield_min": 0.05,       # FCF/EV-Rendite Untergrenze (5%)
    "kbv_max": 3.0,                 # P/B Ratio Obergrenze
    "ccr_min": 0.80,                # FCF/Nettogewinn ≥ 80%
    "roce_min": 0.08,               # EBIT/Capital Employed ≥ 8%
    "zinsdeckung_min": 3.0,         # EBIT/Zinsaufwand ≥ 3×
    "schulden_eq_max": 1.5,         # Schulden/Eigenkapital Obergrenze
    "conviction_gewicht": 0.20,     # 5 Agenten → gleiches Gewicht
    "finanz_sektoren": ["Financial Services", "Utilities", "Real Estate"],
}


# ── Datenhilfs­funktionen ─────────────────────────────────────────────────────

def _ebit_aus_financials(fin) -> Optional[float]:
    """EBIT aus Finanzdaten (yfinance); Fallback auf Operating Income."""
    if fin is None or (hasattr(fin, "empty") and fin.empty):
        return None
    for feld in ("EBIT", "Operating Income"):
        try:
            serie = fin.loc[feld].dropna()
            if not serie.empty:
                return float(serie.iloc[0])
        except KeyError:
            continue
    return None


def _fcf_aus_cashflow(cf) -> Optional[float]:
    """FCF aus Cashflow-Statement (zuverlässiger als info['freeCashflow'])."""
    if cf is None or (hasattr(cf, "empty") and cf.empty):
        return None
    try:
        serie = cf.loc["Free Cash Flow"].dropna()
        if not serie.empty:
            return float(serie.iloc[0])
    except KeyError:
        pass
    return None


def _equity_aus_balance_sheet(bs) -> Optional[float]:
    """Eigenkapital aus Balance Sheet; Fallback auf Common Stock Equity."""
    if bs is None or (hasattr(bs, "empty") and bs.empty):
        return None
    for feld in ("Stockholders Equity", "Common Stock Equity"):
        try:
            serie = bs.loc[feld].dropna()
            if not serie.empty:
                return float(serie.iloc[0])
        except KeyError:
            continue
    return None


def _cash_aus_balance_sheet(bs) -> Optional[float]:
    """Cash & Äquivalente aus Balance Sheet."""
    if bs is None or (hasattr(bs, "empty") and bs.empty):
        return None
    for feld in (
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
    ):
        try:
            serie = bs.loc[feld].dropna()
            if not serie.empty:
                return float(serie.iloc[0])
        except KeyError:
            continue
    return None


def _capital_employed(bs) -> Optional[float]:
    """Capital Employed = Total Assets − Current Liabilities."""
    if bs is None or (hasattr(bs, "empty") and bs.empty):
        return None
    try:
        assets = bs.loc["Total Assets"].dropna()
        curr_liab = bs.loc["Current Liabilities"].dropna()
        if assets.empty or curr_liab.empty:
            return None
        return float(assets.iloc[0]) - float(curr_liab.iloc[0])
    except KeyError:
        return None


# ── Befund-Funktionen ─────────────────────────────────────────────────────────

def _ev_ebit(
    market_cap: Optional[float],
    total_debt: Optional[float],
    cash: Optional[float],
    ebit: Optional[float],
) -> Befund:
    """EV/EBIT: kapitalstrukturneutrale Bewertung. EV = MarketCap + Debt − Cash."""
    label = "EV/EBIT"
    if ebit is None or ebit <= 0:
        return befund_unbestimmt(label, "EBIT nicht verfügbar oder negativ")
    if market_cap is None or market_cap <= 0:
        return befund_unbestimmt(label, "Marktkapitalisierung fehlt")

    ev = market_cap + (total_debt or 0.0) - (cash or 0.0)
    if ev <= 0:
        return befund_unbestimmt(label, "Enterprise Value ≤ 0")

    ratio = ev / ebit
    bestanden = ratio <= SCHWELLEN["ev_ebit_max"]
    return Befund(
        label=label, wert=round(ratio, 2), bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=f"Schwelle ≤ {SCHWELLEN['ev_ebit_max']}",
    )


def _fcf_ev_yield(
    fcf: Optional[float],
    market_cap: Optional[float],
    total_debt: Optional[float],
    cash: Optional[float],
) -> Befund:
    """FCF/EV-Rendite: echter Cashflow auf Gesamtunternehmensbasis."""
    label = "FCF/EV-Rendite"
    if fcf is None:
        return befund_unbestimmt(label, "FCF nicht verfügbar")
    if market_cap is None or market_cap <= 0:
        return befund_unbestimmt(label, "Marktkapitalisierung fehlt")

    ev = market_cap + (total_debt or 0.0) - (cash or 0.0)
    if ev <= 0:
        return befund_unbestimmt(label, "Enterprise Value ≤ 0")

    yield_val = fcf / ev
    bestanden = yield_val >= SCHWELLEN["fcf_ev_yield_min"]
    return Befund(
        label=label, wert=round(yield_val, 4), bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=f"Schwelle ≥ {SCHWELLEN['fcf_ev_yield_min']:.0%}",
    )


def _kbv(pb: Optional[float]) -> Befund:
    """KBV (P/B Ratio): Substanzbewertung."""
    label = "KBV (P/B Ratio)"
    if pb is None or pb <= 0:
        return befund_unbestimmt(label, "KBV nicht verfügbar oder negativ")
    bestanden = pb <= SCHWELLEN["kbv_max"]
    return Befund(
        label=label, wert=round(pb, 2), bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=f"Schwelle ≤ {SCHWELLEN['kbv_max']}",
    )


def _ccr(fcf: Optional[float], fin) -> Befund:
    """Cash Conversion Rate (FCF / Nettogewinn): hohe Ratio → echter Cash-Gewinn, geringe Accruals."""
    label = "Cash Conversion Rate (FCF/Nettogewinn)"
    if fcf is None:
        return befund_unbestimmt(label, "FCF nicht verfügbar")
    if fin is None or (hasattr(fin, "empty") and fin.empty):
        return befund_unbestimmt(label, "Finanzdaten fehlen")
    try:
        net_income_serie = fin.loc["Net Income"].dropna()
        if net_income_serie.empty:
            return befund_unbestimmt(label, "Nettogewinn nicht verfügbar")
        ni = float(net_income_serie.iloc[0])
        if ni <= 0:
            return befund_unbestimmt(label, "Nettogewinn nicht positiv")
        ratio = fcf / ni
        bestanden = ratio >= SCHWELLEN["ccr_min"]
        return Befund(
            label=label, wert=round(ratio, 4), bestanden=bestanden,
            zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
            details=f"Schwelle ≥ {SCHWELLEN['ccr_min']:.0%}",
        )
    except KeyError:
        return befund_unbestimmt(label, "Nettogewinn nicht in Finanzdaten")


def _roce(ebit: Optional[float], bs, sektor: Optional[str]) -> Befund:
    """ROCE (EBIT / Capital Employed): Kapitalrendite.

    Finanz-, Utility- und REIT-Sektoren ausgeschlossen (ROCE nicht sinnvoll anwendbar).
    """
    label = "ROCE (EBIT/Capital Employed)"
    if sektor in SCHWELLEN["finanz_sektoren"]:
        return befund_unbestimmt(label, f"ROCE für Sektor '{sektor}' nicht anwendbar")
    if ebit is None:
        return befund_unbestimmt(label, "EBIT nicht verfügbar")
    ce = _capital_employed(bs)
    if ce is None or ce <= 0:
        return befund_unbestimmt(label, "Capital Employed nicht berechenbar")
    ratio = ebit / ce
    bestanden = ratio >= SCHWELLEN["roce_min"]
    return Befund(
        label=label, wert=round(ratio, 4), bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=f"Schwelle ≥ {SCHWELLEN['roce_min']:.0%}",
    )


def _zinsdeckung(ebit: Optional[float], fin) -> Befund:
    """Zinsdeckungsgrad (EBIT / |Zinsaufwand|).

    Nettoliquidität: Interest Expense ≥ 0 in yfinance → Zinserträge ≥ Zinsaufwand → immer PASSIERT.
    """
    label = "Zinsdeckungsgrad"
    if ebit is None:
        return befund_unbestimmt(label, "EBIT nicht verfügbar")
    if fin is None or (hasattr(fin, "empty") and fin.empty):
        return befund_unbestimmt(label, "Finanzdaten fehlen")
    try:
        interest_serie = fin.loc["Interest Expense"].dropna()
        if interest_serie.empty:
            return befund_unbestimmt(label, "Zinsaufwand nicht verfügbar")
        interest = float(interest_serie.iloc[0])
    except KeyError:
        return befund_unbestimmt(label, "Zinsaufwand nicht in Finanzdaten")

    # yfinance: Interest Expense ist negativ wenn Aufwand, positiv/0 = Nettoliquid
    if interest >= 0:
        return Befund(
            label=label, wert=None, bestanden=True,
            zustand=Zustand.PASSIERT,
            details="Nettoliquid: Zinserträge ≥ Zinsaufwand – kein Schuldzinsrisiko",
        )

    interesse_abs = abs(interest)
    ratio = ebit / interesse_abs
    bestanden = ratio >= SCHWELLEN["zinsdeckung_min"]
    return Befund(
        label=label, wert=round(ratio, 2), bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=f"EBIT/Zinsen = {ratio:.1f}× (Schwelle ≥ {SCHWELLEN['zinsdeckung_min']}×)",
    )


def _schulden(total_debt: Optional[float], total_equity: Optional[float]) -> Befund:
    """Schulden/Eigenkapital: Bilanzstabilität."""
    label = "Schulden/Eigenkapital"
    if total_debt is None or total_equity is None or total_equity <= 0:
        return befund_unbestimmt(label, "Bilanzdaten nicht verfügbar")
    ratio = total_debt / total_equity
    bestanden = ratio <= SCHWELLEN["schulden_eq_max"]
    return Befund(
        label=label, wert=round(ratio, 2), bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=f"Schwelle ≤ {SCHWELLEN['schulden_eq_max']}",
    )


# ── Haupt-API ─────────────────────────────────────────────────────────────────

def analyse(ticker: str) -> List[Befund]:
    """Fundamentale Bewertungs- und Qualitätsanalyse. Gibt immer 7 Befund-Objekte zurück."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        balance_sheet = t.balance_sheet
        cashflow = t.cashflow
        financials = t.financials
    except Exception as exc:
        return [befund_unbestimmt(f"Bewertung {ticker}", f"Datenabruf: {exc}")]

    equity = info.get("totalStockholderEquity") or _equity_aus_balance_sheet(balance_sheet)
    fcf = _fcf_aus_cashflow(cashflow) or info.get("freeCashflow")
    cash = _cash_aus_balance_sheet(balance_sheet)
    ebit = _ebit_aus_financials(financials)
    sektor = info.get("sector")
    market_cap = info.get("marketCap")
    total_debt = info.get("totalDebt")

    return [
        _ev_ebit(market_cap, total_debt, cash, ebit),
        _fcf_ev_yield(fcf, market_cap, total_debt, cash),
        _kbv(info.get("priceToBook")),
        _ccr(fcf, financials),
        _roce(ebit, balance_sheet, sektor),
        _zinsdeckung(ebit, financials),
        _schulden(total_debt, equity),
    ]


def conviction(befunde: List[Befund]) -> float:
    """Score 0–100: Anteil bestandener, bewertbarer Kriterien × 100."""
    bewertbar = [b for b in befunde if b.bestanden is not None]
    if not bewertbar:
        return 0.0
    return round(sum(1 for b in bewertbar if b.bestanden) / len(bewertbar) * 100, 1)

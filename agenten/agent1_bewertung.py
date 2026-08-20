"""Agent 1 – Fundamentale Bewertungsanalyse (Value Investing)."""
from typing import List, Optional

import yfinance as yf

from kern.typen import Befund, Zustand, befund_unbestimmt

SCHWELLEN = {
    "kgv_max": 20.0,           # P/E Ratio Obergrenze
    "kbv_max": 3.0,            # P/B Ratio Obergrenze
    "roe_min": 0.10,           # Return on Equity Untergrenze (10%)
    "fcf_yield_min": 0.03,     # Free Cashflow Rendite Untergrenze (3%)
    "schulden_eq_max": 1.5,    # Schulden/Eigenkapital Obergrenze
    "conviction_gewicht": 0.333, # 3 Agenten → gleiches Gewicht
}


def _kgv(pe: Optional[float]) -> Befund:
    label = "KGV (P/E Ratio)"
    if pe is None or pe <= 0:
        return befund_unbestimmt(label, "KGV nicht verfügbar oder negativ")
    bestanden = pe <= SCHWELLEN["kgv_max"]
    return Befund(
        label=label,
        wert=round(pe, 2),
        bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=f"Schwelle ≤ {SCHWELLEN['kgv_max']}",
    )


def _kbv(pb: Optional[float]) -> Befund:
    label = "KBV (P/B Ratio)"
    if pb is None or pb <= 0:
        return befund_unbestimmt(label, "KBV nicht verfügbar oder negativ")
    bestanden = pb <= SCHWELLEN["kbv_max"]
    return Befund(
        label=label,
        wert=round(pb, 2),
        bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=f"Schwelle ≤ {SCHWELLEN['kbv_max']}",
    )


def _roe(roe: Optional[float]) -> Befund:
    label = "ROE (Eigenkapitalrendite)"
    if roe is None:
        return befund_unbestimmt(label, "ROE nicht verfügbar")
    bestanden = roe >= SCHWELLEN["roe_min"]
    return Befund(
        label=label,
        wert=round(roe, 4),
        bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=f"Schwelle ≥ {SCHWELLEN['roe_min']:.0%}",
    )


def _fcf_yield(fcf: Optional[float], market_cap: Optional[float]) -> Befund:
    label = "FCF-Rendite"
    if fcf is None or market_cap is None or market_cap <= 0:
        return befund_unbestimmt(label, "FCF oder Marktkapitalisierung fehlt")
    yield_val = fcf / market_cap
    bestanden = yield_val >= SCHWELLEN["fcf_yield_min"]
    return Befund(
        label=label,
        wert=round(yield_val, 4),
        bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=f"Schwelle ≥ {SCHWELLEN['fcf_yield_min']:.0%}",
    )


def _schulden(total_debt: Optional[float], total_equity: Optional[float]) -> Befund:
    label = "Schulden/Eigenkapital"
    if total_debt is None or total_equity is None or total_equity <= 0:
        return befund_unbestimmt(label, "Bilanzdaten nicht verfügbar")
    ratio = total_debt / total_equity
    bestanden = ratio <= SCHWELLEN["schulden_eq_max"]
    return Befund(
        label=label,
        wert=round(ratio, 2),
        bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=f"Schwelle ≤ {SCHWELLEN['schulden_eq_max']}",
    )


def _equity_aus_balance_sheet(bs) -> Optional[float]:
    """Eigenkapital aus balance_sheet falls info['totalStockholderEquity'] None liefert."""
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


def _fcf_aus_cashflow(cf) -> Optional[float]:
    """FCF aus Cashflow Statement (zuverlässiger als info['freeCashflow'])."""
    if cf is None or (hasattr(cf, "empty") and cf.empty):
        return None
    try:
        serie = cf.loc["Free Cash Flow"].dropna()
        if not serie.empty:
            return float(serie.iloc[0])
    except KeyError:
        pass
    return None


def analyse(ticker: str) -> List[Befund]:
    """Fundamentale Bewertungsanalyse. Gibt immer Befund-Objekte zurück."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        balance_sheet = t.balance_sheet
        cashflow = t.cashflow
    except Exception as exc:
        return [befund_unbestimmt(f"Bewertung {ticker}", f"Datenabruf: {exc}")]

    # Eigenkapital: info-Feld ist in neuerer yfinance-Version oft None → Balance Sheet
    equity = info.get("totalStockholderEquity") or _equity_aus_balance_sheet(balance_sheet)

    # FCF: info-Feld kann falsche Werte liefern → Cashflow Statement bevorzugen
    fcf = _fcf_aus_cashflow(cashflow) or info.get("freeCashflow")

    return [
        _kgv(info.get("trailingPE")),
        _kbv(info.get("priceToBook")),
        _roe(info.get("returnOnEquity")),
        _fcf_yield(fcf, info.get("marketCap")),
        _schulden(info.get("totalDebt"), equity),
    ]


def conviction(befunde: List[Befund]) -> float:
    """Score 0–100: Anteil bestandener, bewertbarer Kriterien × 100."""
    bewertbar = [b for b in befunde if b.bestanden is not None]
    if not bewertbar:
        return 0.0
    return round(sum(1 for b in bewertbar if b.bestanden) / len(bewertbar) * 100, 1)

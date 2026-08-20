"""
Ursachen-Klassifikator für Kursrückgänge.

Klassifiziert den wahrscheinlichen Grund eines Drawdowns anhand:
- News-Schlagzeilen (Keyword-Matching)
- Sektor-Relativperformance (Ticker vs. Sektor-ETF)
- Earnings-Proximity (Tage bis zum nächsten Quartalsbericht)
"""
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf

EARNINGS_VETO_TAGE = 5

SEKTOR_ETF = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
}


class Ursache(Enum):
    MACRO = "MACRO"
    SECTOR = "SECTOR"
    GUIDANCE_INTACT = "GUIDANCE_INTACT"
    GUIDANCE_LOWERED = "GUIDANCE_LOWERED"
    GOVERNANCE_LEGAL = "GOVERNANCE_LEGAL"
    MA = "MA"
    REGULATORY = "REGULATORY"
    UNKNOWN = "UNKNOWN"


@dataclass
class DipDiagnose:
    ursache: Ursache
    sektor_delta: Optional[float]
    tage_bis_earnings: Optional[int]
    veto: bool
    veto_grund: str = ""
    sektor_etf: str = ""


_KEYWORDS: dict = {
    Ursache.GOVERNANCE_LEGAL: [
        "fraud", "lawsuit", "sec investigation", "criminal", "scandal",
        "bribery", "corruption", "accounting irregularities", "restatement",
        "betrug", "klage", "ermittlung",
    ],
    Ursache.REGULATORY: [
        "antitrust", "fda", "penalty", "sanctions", "tariff", "compliance",
        "geldstrafe", "regulierung",
    ],
    Ursache.MA: [
        "merger", "acquisition", "buyout", "takeover", "bid",
        "übernahme", "fusion",
    ],
    Ursache.GUIDANCE_LOWERED: [
        "guidance cut", "lowered guidance", "profit warning",
        "misses estimates", "disappoints", "downgrade",
    ],
    Ursache.GUIDANCE_INTACT: [
        "beats expectations", "raised guidance", "record results",
        "exceeds", "outperformed",
    ],
    Ursache.MACRO: [
        "federal reserve", "interest rate", "inflation", "recession",
        "market sell-off", "yield curve", "gdp slowdown",
        "zinsen", "wirtschaft",
    ],
}

# Priority: GOVERNANCE_LEGAL wins over all others if detected
_PRIORITY_ORDER = [
    Ursache.GOVERNANCE_LEGAL,
    Ursache.REGULATORY,
    Ursache.MA,
    Ursache.GUIDANCE_LOWERED,
    Ursache.GUIDANCE_INTACT,
    Ursache.MACRO,
]


def _klassifiziere_headline(text: str) -> Optional[Ursache]:
    """Klassifiziert einen News-Text per Keyword-Matching. Gibt None zurück bei keinem Treffer."""
    t = text.lower()
    for ursache in _PRIORITY_ORDER:
        if any(kw in t for kw in _KEYWORDS[ursache]):
            return ursache
    return None


def _news_ursache(ticker_obj) -> Ursache:
    """Aggregiert News-Klassifikation über bis zu 20 Schlagzeilen."""
    try:
        news = ticker_obj.news or []
    except Exception:
        return Ursache.UNKNOWN

    if not news:
        return Ursache.UNKNOWN

    stimmen: dict = {}
    for item in news[:20]:
        text = item.get("title", "") + " " + item.get("summary", "")
        ursache = _klassifiziere_headline(text)
        if ursache:
            stimmen[ursache] = stimmen.get(ursache, 0) + 1

    if not stimmen:
        return Ursache.UNKNOWN

    if Ursache.GOVERNANCE_LEGAL in stimmen:
        return Ursache.GOVERNANCE_LEGAL

    return max(stimmen, key=lambda k: stimmen[k])


def _berechne_sektor_delta(
    ticker_sym: str, sektor: str, fenster_tage: int = 30
) -> Tuple[Optional[float], str]:
    """Rendite-Differenz: Ticker minus Sektor-ETF im Vergleichszeitraum.

    Negativ = Ticker lief schlechter als Sektor (idiosynkratisch).
    """
    etf = SEKTOR_ETF.get(sektor, "")
    if not etf:
        return None, ""

    try:
        period = f"{max(fenster_tage, 30)}d"
        hist_t = yf.download(ticker_sym, period=period, progress=False, auto_adjust=True)
        hist_e = yf.download(etf, period=period, progress=False, auto_adjust=True)

        if hist_t.empty or hist_e.empty:
            return None, etf

        close_t = hist_t["Close"].squeeze()
        close_e = hist_e["Close"].squeeze()
        if hasattr(close_t, "columns"):
            close_t = close_t.iloc[:, 0]
        if hasattr(close_e, "columns"):
            close_e = close_e.iloc[:, 0]
        close_t = close_t.dropna()
        close_e = close_e.dropna()

        if len(close_t) < 2 or len(close_e) < 2:
            return None, etf

        ret_t = float(close_t.iloc[-1] / close_t.iloc[0] - 1)
        ret_e = float(close_e.iloc[-1] / close_e.iloc[0] - 1)
        return round(ret_t - ret_e, 4), etf
    except Exception:
        return None, etf


def _tage_bis_earnings(ticker_obj) -> Optional[int]:
    """Tage bis zum nächsten Earnings-Termin. 0 wenn Termin bereits vergangen."""
    try:
        cal = ticker_obj.calendar
        if cal is None:
            return None

        earnings_date = None
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
            if dates:
                earnings_date = dates[0] if isinstance(dates, (list, tuple)) else dates
        elif isinstance(cal, pd.DataFrame):
            for col in cal.columns:
                if "Earnings" in str(col):
                    vals = cal[col].dropna()
                    if not vals.empty:
                        earnings_date = vals.iloc[0]
                        break
            if earnings_date is None and "Earnings Date" in cal.index:
                earnings_date = cal.loc["Earnings Date"].iloc[0]

        if earnings_date is None:
            return None

        ed = pd.Timestamp(earnings_date).date()
        delta = (ed - date.today()).days
        return max(delta, 0)
    except Exception:
        return None


def klassifiziere(ticker: str, fenster_tage: int = 30) -> DipDiagnose:
    """Klassifiziert die wahrscheinliche Ursache eines Kursrückgangs für einen Ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        sektor = info.get("sector", "")

        ursache = _news_ursache(t)
        delta, etf_sym = _berechne_sektor_delta(ticker, sektor, fenster_tage)
        tage = _tage_bis_earnings(t)

        veto = False
        veto_grund = ""
        if ursache == Ursache.GOVERNANCE_LEGAL:
            veto = True
            veto_grund = "Governance/Legal-Risiko erkannt"
        elif tage is not None and tage < EARNINGS_VETO_TAGE:
            veto = True
            veto_grund = f"Earnings in {tage} Tag(en) – zu nah"

        return DipDiagnose(
            ursache=ursache,
            sektor_delta=delta,
            tage_bis_earnings=tage,
            veto=veto,
            veto_grund=veto_grund,
            sektor_etf=etf_sym,
        )
    except Exception as exc:
        return DipDiagnose(
            ursache=Ursache.UNKNOWN,
            sektor_delta=None,
            tage_bis_earnings=None,
            veto=False,
            veto_grund=f"Klassifikationsfehler: {exc}",
        )

"""
Agent 5 – Insider/Sentiment.

Beantwortet: Zeigt das Smart Money Kaufinteresse?

1. Insider-Netto-Aktivität   – Käufe > Verkäufe in letzten 90 Tagen
2. Short Interest             – % Float leerverkauft (niedrig = positiv)
3. Analysten-Konsens          – ≥ 50% Buy/Strong Buy empfehlen
4. Institutionelle Beteiligung – 30–98% institutionell gehalten
"""
from typing import List, Optional

import pandas as pd
import yfinance as yf

from kern.typen import Befund, Zustand, befund_unbestimmt

SCHWELLEN = {
    "insider_netto_kauf_min_tage": 90,   # Betrachtungsfenster Insider-Käufe
    "short_interest_max": 0.10,           # ≤ 10% Float leerverkauft
    "analyst_buy_ratio_min": 0.50,        # ≥ 50% Buy/StrongBuy-Empfehlungen
    "institutionell_min": 0.30,           # ≥ 30% institutionell gehalten
    "institutionell_max": 0.98,           # ≤ 98% – Streubesitz muss vorhanden sein
    "conviction_gewicht": 0.20,           # 5 Agenten → gleiches Gewicht
}


# ── Befund-Funktionen ─────────────────────────────────────────────────────────

def _insider_netto(transactions) -> Befund:
    """Netto-Insider-Aktivität: Käufe − Verkäufe (Aktien) in letzten 90 Tagen."""
    label = "Insider-Netto-Aktivität (90 Tage)"

    if transactions is None:
        return befund_unbestimmt(label, "Keine Insider-Transaktionsdaten")

    if hasattr(transactions, "empty") and transactions.empty:
        return befund_unbestimmt(label, "Keine Insider-Transaktionsdaten")

    try:
        cutoff = pd.Timestamp.now()
        if transactions.index.tz is not None:
            cutoff = cutoff.tz_localize(transactions.index.tz)
        cutoff = cutoff - pd.Timedelta(days=SCHWELLEN["insider_netto_kauf_min_tage"])
        recent = transactions[transactions.index >= cutoff]

        if recent.empty:
            return befund_unbestimmt(label, "Keine Transaktionen in letzten 90 Tagen")

        # yfinance: Spalte "Insider Trading" enthält Typ (Purchase/Sale/…)
        text_col = None
        for col in ("Insider Trading", "Transaction", "Type"):
            if col in recent.columns:
                text_col = col
                break

        if text_col is None:
            return befund_unbestimmt(label, "Transaktionstyp nicht identifizierbar")

        kaufe, verkauefe = 0.0, 0.0
        for _, row in recent.iterrows():
            typ = str(row.get(text_col, "")).lower()
            shares = abs(row.get("Shares", 0) or 0)
            if any(k in typ for k in ("purchase", "buy", "acquisition")):
                kaufe += shares
            elif any(k in typ for k in ("sale", "sell", "sold")):
                verkauefe += shares

        if kaufe == 0 and verkauefe == 0:
            return befund_unbestimmt(label, "Keine klassifizierbaren Transaktionen")

        netto = kaufe - verkauefe
        bestanden = netto > 0
        return Befund(
            label=label,
            wert=round(netto),
            bestanden=bestanden,
            zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
            details=(
                f"Käufe: {kaufe:,.0f} Aktien, Verkäufe: {verkauefe:,.0f} Aktien "
                f"(Netto: {netto:+,.0f})"
            ),
        )

    except Exception as exc:
        return befund_unbestimmt(label, f"Auswertung fehlgeschlagen: {exc}")


def _short_interest(short_pct: Optional[float]) -> Befund:
    """Short Interest als % des Float – niedriger Wert = Bären uninteressiert."""
    label = "Short Interest (% Float)"

    if short_pct is None:
        return befund_unbestimmt(label, "Short Interest nicht verfügbar")

    bestanden = short_pct <= SCHWELLEN["short_interest_max"]
    return Befund(
        label=label,
        wert=round(short_pct, 4),
        bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=(
            f"{short_pct:.1%} Float leerverkauft "
            f"(Schwelle ≤ {SCHWELLEN['short_interest_max']:.0%})"
        ),
    )


def _analyst_konsens(recommendations) -> Befund:
    """Analysten-Konsens: Anteil Buy/StrongBuy an allen Empfehlungen."""
    label = "Analysten-Konsens"

    if recommendations is None:
        return befund_unbestimmt(label, "Keine Analystenempfehlungen verfügbar")

    if hasattr(recommendations, "empty") and recommendations.empty:
        return befund_unbestimmt(label, "Keine Analystenempfehlungen verfügbar")

    try:
        # recommendations hat Spalten: period, strongBuy, buy, hold, sell, strongSell
        if "period" in recommendations.columns:
            current = recommendations[recommendations["period"] == "0m"]
            row_src = current if not current.empty else recommendations.iloc[[0]]
        else:
            row_src = recommendations.iloc[[0]]

        row = row_src.iloc[0]
        strong_buy = int(row.get("strongBuy", 0) or 0)
        buy = int(row.get("buy", 0) or 0)
        hold = int(row.get("hold", 0) or 0)
        sell = int(row.get("sell", 0) or 0)
        strong_sell = int(row.get("strongSell", 0) or 0)

        total = strong_buy + buy + hold + sell + strong_sell
        if total == 0:
            return befund_unbestimmt(label, "Keine Empfehlungen vorhanden")

        buy_ratio = (strong_buy + buy) / total
        bestanden = buy_ratio >= SCHWELLEN["analyst_buy_ratio_min"]
        return Befund(
            label=label,
            wert=round(buy_ratio, 4),
            bestanden=bestanden,
            zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
            details=(
                f"{buy_ratio:.0%} Buy/StrongBuy "
                f"({strong_buy + buy}/{total} Analysten, "
                f"Schwelle ≥ {SCHWELLEN['analyst_buy_ratio_min']:.0%})"
            ),
        )

    except Exception as exc:
        return befund_unbestimmt(label, f"Auswertung fehlgeschlagen: {exc}")


def _institutionelle_beteiligung(held_pct: Optional[float]) -> Befund:
    """Institutionelle Beteiligung: Smart Money präsent, aber noch Streubesitz-Spielraum."""
    label = "Institutionelle Beteiligung"

    if held_pct is None:
        return befund_unbestimmt(label, "Institutionelle Beteiligung nicht verfügbar")

    if held_pct < SCHWELLEN["institutionell_min"]:
        return Befund(
            label=label,
            wert=round(held_pct, 4),
            bestanden=False,
            zustand=Zustand.NICHT_PASSIERT,
            details=(
                f"Nur {held_pct:.0%} institutionell "
                f"(Schwelle ≥ {SCHWELLEN['institutionell_min']:.0%})"
            ),
        )

    if held_pct > SCHWELLEN["institutionell_max"]:
        return Befund(
            label=label,
            wert=round(held_pct, 4),
            bestanden=False,
            zustand=Zustand.NICHT_PASSIERT,
            details=(
                f"{held_pct:.0%} institutionell – kaum Streubesitz vorhanden, "
                "Liquiditätsrisiko"
            ),
        )

    return Befund(
        label=label,
        wert=round(held_pct, 4),
        bestanden=True,
        zustand=Zustand.PASSIERT,
        details=(
            f"{held_pct:.0%} institutionell gehalten "
            f"({SCHWELLEN['institutionell_min']:.0%}–{SCHWELLEN['institutionell_max']:.0%})"
        ),
    )


# ── Haupt-API ─────────────────────────────────────────────────────────────────

def analyse(ticker: str) -> List[Befund]:
    """Insider/Sentiment-Analyse. Gibt immer 4 Befund-Objekte zurück."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        insider_txn = t.insider_transactions
        recommendations = t.recommendations
    except Exception as exc:
        lbl = f"Insider/Sentiment {ticker}"
        return [befund_unbestimmt(lbl, f"Datenabruf: {exc}") for _ in range(4)]

    return [
        _insider_netto(insider_txn),
        _short_interest(info.get("shortPercentOfFloat")),
        _analyst_konsens(recommendations),
        _institutionelle_beteiligung(info.get("heldPercentInstitutions")),
    ]


def conviction(befunde: List[Befund]) -> float:
    """Score 0–100: Anteil bestandener, bewertbarer Kriterien × 100."""
    bewertbar = [b for b in befunde if b.bestanden is not None]
    if not bewertbar:
        return 0.0
    return round(sum(1 for b in bewertbar if b.bestanden) / len(bewertbar) * 100, 1)

"""
Agent 3 – Qualitätsanalyse (Earnings Quality & Business Durability).

Prüft, ob ein Unternehmen dauerhaft profitabel wirtschaftet:
1. FCF-Konsistenz: War der Free Cashflow über mehrere Jahre positiv?
2. Gewinnqualität: Wie viel des Nettogewinns landet als echter Cash?
3. Margen-Stabilität: Ist die Bruttomarge konstant (Preissetzungsmacht)?
4. ROA: Arbeitet das Kapital effizient?
"""
from typing import List, Optional

import pandas as pd
import yfinance as yf

from kern.typen import Befund, Zustand, befund_unbestimmt

SCHWELLEN = {
    "fcf_positiv_jahre_min": 3,     # FCF mind. 3 Jahre in Folge positiv
    "gewinnqualitaet_min": 0.70,    # FCF / Nettogewinn ≥ 70% (Accruals gering)
    "marge_std_max": 0.05,          # Bruttomarge-Standardabw. ≤ 5 PP (stabil)
    "roa_min": 0.05,                # ROA ≥ 5%
    "conviction_gewicht": 0.333,    # 3 Agenten → gleiches Gewicht
}


def _fcf_konsistenz(cashflow: pd.DataFrame) -> Befund:
    """FCF war in den letzten N Jahren durchgehend positiv."""
    label = "FCF-Konsistenz"
    try:
        fcf = cashflow.loc["Free Cash Flow"].dropna()
    except (KeyError, AttributeError, TypeError):
        return befund_unbestimmt(label, "Free Cash Flow nicht in Daten")

    if len(fcf) < SCHWELLEN["fcf_positiv_jahre_min"]:
        return befund_unbestimmt(
            label,
            f"Nur {len(fcf)} Jahre Daten (mind. {SCHWELLEN['fcf_positiv_jahre_min']} nötig)",
        )

    positiv = int((fcf > 0).sum())
    alle_positiv = positiv == len(fcf)
    return Befund(
        label=label,
        wert=positiv,
        bestanden=alle_positiv,
        zustand=Zustand.PASSIERT if alle_positiv else Zustand.NICHT_PASSIERT,
        details=f"FCF positiv in {positiv}/{len(fcf)} Jahren",
    )


def _gewinnqualitaet(cashflow: pd.DataFrame, financials: pd.DataFrame) -> Befund:
    """FCF / Nettogewinn: hohe Ratio → kaum Accruals, echter Cash-Gewinn."""
    label = "Gewinnqualität (FCF / Nettogewinn)"
    try:
        fcf = cashflow.loc["Free Cash Flow"].dropna()
        net = financials.loc["Net Income"].dropna()
    except (KeyError, AttributeError, TypeError):
        return befund_unbestimmt(label, "FCF oder Nettogewinn nicht verfügbar")

    idx = fcf.index.intersection(net.index)
    valid = [i for i in idx if net.loc[i] > 0]  # nur Jahre mit positivem Gewinn
    if len(valid) < 2:
        return befund_unbestimmt(label, "Weniger als 2 Jahre mit positivem Nettogewinn")

    ratios = fcf.loc[valid] / net.loc[valid]
    ratio_mean = float(ratios.mean())
    bestanden = ratio_mean >= SCHWELLEN["gewinnqualitaet_min"]
    return Befund(
        label=label,
        wert=round(ratio_mean, 4),
        bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=f"Ø FCF/Nettogewinn: {ratio_mean:.0%} (Schwelle ≥ {SCHWELLEN['gewinnqualitaet_min']:.0%})",
    )


def _margen_stabilitaet(financials: pd.DataFrame) -> Befund:
    """Bruttomarge stabil über mehrere Jahre → Preissetzungsmacht vorhanden."""
    label = "Bruttomarge-Stabilität"
    try:
        gross = financials.loc["Gross Profit"].dropna()
        revenue = financials.loc["Total Revenue"].dropna()
    except (KeyError, AttributeError, TypeError):
        return befund_unbestimmt(label, "Gewinn-/Umsatzdaten nicht verfügbar")

    idx = gross.index.intersection(revenue.index)
    valid = [i for i in idx if revenue.loc[i] > 0]
    if len(valid) < 2:
        return befund_unbestimmt(label, "Zu wenig historische Daten")

    margins = gross.loc[valid] / revenue.loc[valid]
    std = float(margins.std())
    mean = float(margins.mean())
    bestanden = std <= SCHWELLEN["marge_std_max"]
    return Befund(
        label=label,
        wert=round(mean, 4),
        bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=(
            f"Bruttomarge Ø {mean:.1%} ± {std:.1%} "
            f"(Schwelle: σ ≤ {SCHWELLEN['marge_std_max']:.0%})"
        ),
    )


def _roa(roa_wert: Optional[float]) -> Befund:
    """Return on Assets: Effizienz der gesamten Aktivabasis."""
    label = "ROA (Gesamtkapitalrendite)"
    if roa_wert is None:
        return befund_unbestimmt(label, "ROA nicht verfügbar")
    bestanden = roa_wert >= SCHWELLEN["roa_min"]
    return Befund(
        label=label,
        wert=round(roa_wert, 4),
        bestanden=bestanden,
        zustand=Zustand.PASSIERT if bestanden else Zustand.NICHT_PASSIERT,
        details=f"Schwelle ≥ {SCHWELLEN['roa_min']:.0%}",
    )


def analyse(ticker: str) -> List[Befund]:
    """Qualitätsanalyse. Gibt immer Befund-Objekte zurück."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        cashflow = t.cashflow
        financials = t.financials
    except Exception as exc:
        return [befund_unbestimmt(f"Qualität {ticker}", f"Datenabruf: {exc}")]

    if cashflow is None or cashflow.empty:
        cashflow = pd.DataFrame()
    if financials is None or financials.empty:
        financials = pd.DataFrame()

    return [
        _fcf_konsistenz(cashflow),
        _gewinnqualitaet(cashflow, financials),
        _margen_stabilitaet(financials),
        _roa(info.get("returnOnAssets")),
    ]


def conviction(befunde: List[Befund]) -> float:
    """Score 0–100: Anteil bestandener, bewertbarer Kriterien × 100."""
    bewertbar = [b for b in befunde if b.bestanden is not None]
    if not bewertbar:
        return 0.0
    return round(sum(1 for b in bewertbar if b.bestanden) / len(bewertbar) * 100, 1)

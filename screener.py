#!/usr/bin/env python3
"""AgentTrade – Value Investing Screener CLI."""
import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import List

import agenten.agent1_bewertung as a1
import agenten.agent2_drawdown as a2
import agenten.agent3_dip as a3
import agenten.agent4_wachstum as a4
import agenten.agent5_insider as a5
import agenten.agent6_kapitalrueckgabe as a6
import agenten.agent7_momentum as a7
import agenten.agent8_bilanzqualitaet as a8
from kern import version
from kern.typen import Befund, Zustand

ERGEBNISSE_DIR = Path("ergebnisse")


def _gesamtconviction(
    befunde_a1: List[Befund],
    befunde_a2: List[Befund],
    befunde_a3: List[Befund],
    befunde_a4: List[Befund],
    befunde_a5: List[Befund],
    befunde_a6: List[Befund],
    befunde_a7: List[Befund],
    befunde_a8: List[Befund],
) -> float:
    agenten = [a1, a2, a3, a4, a5, a6, a7, a8]
    befunde_liste = [befunde_a1, befunde_a2, befunde_a3, befunde_a4,
                     befunde_a5, befunde_a6, befunde_a7, befunde_a8]
    gewichte = [ax.SCHWELLEN["conviction_gewicht"] for ax in agenten]
    scores = [ax.conviction(bx) for ax, bx in zip(agenten, befunde_liste)]
    gesamt_g = sum(gewichte)
    return round(sum(g * c for g, c in zip(gewichte, scores)) / gesamt_g, 1)


def screene(ticker: str) -> dict:
    befunde_a1 = a1.analyse(ticker)
    befunde_a2 = a2.analyse(ticker)
    befunde_a3 = a3.analyse(ticker)
    befunde_a4 = a4.analyse(ticker)
    befunde_a5 = a5.analyse(ticker)
    befunde_a6 = a6.analyse(ticker)
    befunde_a7 = a7.analyse(ticker)
    befunde_a8 = a8.analyse(ticker)
    return {
        "ticker": ticker,
        "datum": str(date.today()),
        "version": version.versionstempel(),
        "conviction": _gesamtconviction(
            befunde_a1, befunde_a2, befunde_a3, befunde_a4,
            befunde_a5, befunde_a6, befunde_a7, befunde_a8,
        ),
        "agent1": [b.as_dict() for b in befunde_a1],
        "agent2": [b.as_dict() for b in befunde_a2],
        "agent3": [b.as_dict() for b in befunde_a3],
        "agent4": [b.as_dict() for b in befunde_a4],
        "agent5": [b.as_dict() for b in befunde_a5],
        "agent6": [b.as_dict() for b in befunde_a6],
        "agent7": [b.as_dict() for b in befunde_a7],
        "agent8": [b.as_dict() for b in befunde_a8],
    }


def _zeige_befunde(titel: str, befund_dicts: list) -> None:
    print(f"\n[{titel}]")
    for d in befund_dicts:
        b = Befund.from_dict(d)
        print(f"  {b}")
        if b.details:
            print(f"     → {b.details}")


def zeige_ergebnis(e: dict) -> None:
    conv = e["conviction"]
    stars = "★" * int(conv // 20) + "☆" * (5 - int(conv // 20))
    print(f"\n{'='*62}")
    print(f"  {e['ticker']:8s}  Conviction: {conv:5.1f}/100  {stars}")
    print(f"  Datum: {e['datum']}   Version: {e['version']}")
    print(f"{'='*62}")
    _zeige_befunde("Agent 1 – Bewertung", e["agent1"])
    _zeige_befunde("Agent 2 – Drawdown", e["agent2"])
    _zeige_befunde("Agent 3 – Dip-Diagnose", e.get("agent3", []))
    _zeige_befunde("Agent 4 – Wachstum", e.get("agent4", []))
    _zeige_befunde("Agent 5 – Insider/Sentiment", e.get("agent5", []))
    _zeige_befunde("Agent 6 – Kapitalrückgabe", e.get("agent6", []))
    _zeige_befunde("Agent 7 – Momentum", e.get("agent7", []))
    _zeige_befunde("Agent 8 – Bilanzqualität", e.get("agent8", []))


def speichere(e: dict) -> None:
    ERGEBNISSE_DIR.mkdir(exist_ok=True)
    path = ERGEBNISSE_DIR / f"{e['ticker']}_{e['datum']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(e, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n  → Gespeichert: {path}")


def review(ticker: str) -> None:
    dateien = sorted(ERGEBNISSE_DIR.glob(f"{ticker}_*.json"))
    if not dateien:
        print(f"Keine gespeicherten Ergebnisse für {ticker}.")
        return
    aktuelle_v = version.versionstempel()
    print(f"\n{'='*62}")
    print(f"  REVIEW: {ticker}  ({len(dateien)} Einträge)")
    print(f"  Aktuelle Version: {aktuelle_v}")
    print(f"{'='*62}")
    for p in dateien:
        with open(p, encoding="utf-8") as f:
            e = json.load(f)
        v_flag = "" if e.get("version") == aktuelle_v else "  ⚠ ANDERE VERSION"
        print(f"  {e['datum']}  Conviction {e['conviction']:5.1f}  [{e['version']}]{v_flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentTrade – Value Investing Screener")
    parser.add_argument("tickers", nargs="*", help="Ticker-Symbole (z.B. AAPL MSFT)")
    parser.add_argument("--review", action="store_true", help="Historische Scores anzeigen")
    parser.add_argument("--speichern", action="store_true", help="Ergebnisse als JSON speichern")
    args = parser.parse_args()

    if not args.tickers:
        parser.print_help()
        sys.exit(1)

    for ticker in [t.upper() for t in args.tickers]:
        if args.review:
            review(ticker)
        else:
            e = screene(ticker)
            zeige_ergebnis(e)
            if args.speichern:
                speichere(e)


if __name__ == "__main__":
    main()

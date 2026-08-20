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
from kern import version
from kern.typen import Befund, Zustand

ERGEBNISSE_DIR = Path("ergebnisse")


def _gesamtconviction(
    befunde_a1: List[Befund],
    befunde_a2: List[Befund],
    befunde_a3: List[Befund],
) -> float:
    g1 = a1.SCHWELLEN["conviction_gewicht"]
    g2 = a2.SCHWELLEN["conviction_gewicht"]
    g3 = a3.SCHWELLEN["conviction_gewicht"]
    gesamt_g = g1 + g2 + g3
    c1 = a1.conviction(befunde_a1)
    c2 = a2.conviction(befunde_a2)
    c3 = a3.conviction(befunde_a3)
    return round((g1 * c1 + g2 * c2 + g3 * c3) / gesamt_g, 1)


def screene(ticker: str) -> dict:
    befunde_a1 = a1.analyse(ticker)
    befunde_a2 = a2.analyse(ticker)
    befunde_a3 = a3.analyse(ticker)
    return {
        "ticker": ticker,
        "datum": str(date.today()),
        "version": version.versionstempel(),
        "conviction": _gesamtconviction(befunde_a1, befunde_a2, befunde_a3),
        "agent1": [b.as_dict() for b in befunde_a1],
        "agent2": [b.as_dict() for b in befunde_a2],
        "agent3": [b.as_dict() for b in befunde_a3],
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

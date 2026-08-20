#!/usr/bin/env python3
"""AgentTrade – Value Investing Screener CLI."""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import List, Optional

import agenten.agent1_bewertung as a1
import agenten.agent2_drawdown as a2
import agenten.agent3_dip as a3
import agenten.agent4_wachstum as a4
import agenten.agent5_insider as a5
import agenten.agent6_kapitalrueckgabe as a6
import agenten.agent7_momentum as a7
import agenten.agent8_bilanzqualitaet as a8
from kern import version
from kern.typen import Befund

_AGENTEN = [a1, a2, a3, a4, a5, a6, a7, a8]
_AGENTEN_NAMEN = [
    "Agent 1 – Bewertung",
    "Agent 2 – Drawdown",
    "Agent 3 – Dip-Diagnose",
    "Agent 4 – Wachstum",
    "Agent 5 – Insider/Sentiment",
    "Agent 6 – Kapitalrückgabe",
    "Agent 7 – Momentum",
    "Agent 8 – Bilanzqualität",
]
ERGEBNISSE_DIR = Path("ergebnisse")


# ── Conviction ────────────────────────────────────────────────────────────────

def _gesamtconviction(alle_befunde: List[List[Befund]]) -> float:
    gewichte = [ax.SCHWELLEN["conviction_gewicht"] for ax in _AGENTEN]
    scores = [ax.conviction(bx) for ax, bx in zip(_AGENTEN, alle_befunde)]
    gesamt_g = sum(gewichte)
    return round(sum(g * c for g, c in zip(gewichte, scores)) / gesamt_g, 1)


# ── Screening ─────────────────────────────────────────────────────────────────

def screene(ticker: str) -> dict:
    """Alle 8 Agenten parallel ausführen und Ergebnis-Dict zurückgeben."""
    alle_befunde: List[Optional[List[Befund]]] = [None] * len(_AGENTEN)

    with ThreadPoolExecutor(max_workers=len(_AGENTEN)) as pool:
        future_to_idx = {
            pool.submit(ax.analyse, ticker): i
            for i, ax in enumerate(_AGENTEN)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                alle_befunde[idx] = future.result()
            except Exception as exc:
                from kern.typen import befund_unbestimmt
                alle_befunde[idx] = [befund_unbestimmt(f"Agent {idx+1}", str(exc))]

    return {
        "ticker": ticker,
        "datum": str(date.today()),
        "version": version.versionstempel(),
        "conviction": _gesamtconviction(alle_befunde),
        **{f"agent{i+1}": [b.as_dict() for b in bx]
           for i, bx in enumerate(alle_befunde)},
    }


# ── Ausgabe ───────────────────────────────────────────────────────────────────

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
    for i, name in enumerate(_AGENTEN_NAMEN):
        _zeige_befunde(name, e.get(f"agent{i+1}", []))


def zeige_ranking(ergebnisse: List[dict], top: Optional[int] = None) -> None:
    """Sortierte Ranking-Tabelle über alle gescreenten Ticker."""
    sortiert = sorted(ergebnisse, key=lambda e: e["conviction"], reverse=True)
    if top:
        sortiert = sortiert[:top]

    v = sortiert[0]["version"] if sortiert else version.versionstempel()
    n = len(ergebnisse)
    breite = 54
    print(f"\n{'='*breite}")
    print(f"  RANKING  –  {n} Kandidat{'en' if n != 1 else ''}  ({v}, {date.today()})")
    print(f"  {'─'*50}")
    print(f"  {'#':>3}  {'Ticker':<10}  {'Conviction':>10}  Sterne")
    print(f"  {'─'*50}")
    for rang, e in enumerate(sortiert, start=1):
        conv = e["conviction"]
        sterne = "★" * int(conv // 20) + "☆" * (5 - int(conv // 20))
        print(f"  {rang:>3}  {e['ticker']:<10}  {conv:>9.1f}  {sterne}")
    print(f"{'='*breite}")


# ── Persistenz ────────────────────────────────────────────────────────────────

def speichere(e: dict) -> None:
    ERGEBNISSE_DIR.mkdir(exist_ok=True)
    path = ERGEBNISSE_DIR / f"{e['ticker']}_{e['datum']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(e, f, ensure_ascii=False, default=str, indent=2)
    print(f"  → Gespeichert: {path}")


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


# ── Watchlist ─────────────────────────────────────────────────────────────────

def lese_watchlist(pfad: Path) -> List[str]:
    """Watchlist lesen: eine Zeile pro Ticker, # = Kommentar."""
    tickers = []
    for zeile in pfad.read_text(encoding="utf-8").splitlines():
        bereinigt = zeile.split("#")[0].strip().upper()
        if bereinigt:
            tickers.append(bereinigt)
    return tickers


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AgentTrade – Value Investing Screener")
    parser.add_argument("tickers", nargs="*", help="Ticker-Symbole (z.B. AAPL MSFT)")
    parser.add_argument("--watchlist", metavar="DATEI",
                        help="Watchlist-Datei (ein Ticker pro Zeile)")
    parser.add_argument("--top", type=int, metavar="N",
                        help="Nur die Top-N nach Conviction anzeigen")
    parser.add_argument("--review", action="store_true",
                        help="Historische Scores anzeigen")
    parser.add_argument("--speichern", action="store_true",
                        help="Ergebnisse als JSON speichern")
    args = parser.parse_args()

    # Ticker-Liste zusammenstellen
    tickers: List[str] = [t.upper() for t in args.tickers]
    if args.watchlist:
        pfad = Path(args.watchlist)
        if not pfad.exists():
            print(f"Fehler: Watchlist-Datei '{pfad}' nicht gefunden.", file=sys.stderr)
            sys.exit(1)
        tickers = lese_watchlist(pfad)
        if not tickers:
            print("Watchlist ist leer.", file=sys.stderr)
            sys.exit(1)

    if not tickers:
        parser.print_help()
        sys.exit(1)

    # Review-Modus
    if args.review:
        for ticker in tickers:
            review(ticker)
        return

    # Screening
    ergebnisse: List[dict] = []
    for ticker in tickers:
        if len(tickers) > 1:
            print(f"  Screene {ticker} …", end=" ", flush=True)
        e = screene(ticker)
        ergebnisse.append(e)
        if len(tickers) > 1:
            print(f"Conviction {e['conviction']:.1f}")
        else:
            zeige_ergebnis(e)
        if args.speichern:
            speichere(e)

    # Ranking bei mehreren Tickers
    if len(ergebnisse) > 1:
        zeige_ranking(ergebnisse, top=args.top)


if __name__ == "__main__":
    main()

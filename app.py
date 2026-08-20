"""AgentTrade – Streamlit Web-Interface."""
from pathlib import Path

import pandas as pd
import streamlit as st

from kern.typen import Befund
from screener import lese_watchlist, screene
import agenten.agent1_bewertung as a1
import agenten.agent2_drawdown as a2
import agenten.agent3_dip as a3
import agenten.agent4_wachstum as a4
import agenten.agent5_insider as a5
import agenten.agent6_kapitalrueckgabe as a6
import agenten.agent7_momentum as a7
import agenten.agent8_bilanzqualitaet as a8

_AGENTEN = [a1, a2, a3, a4, a5, a6, a7, a8]
_AGENTEN_NAMEN = [
    "Bewertung",
    "Drawdown",
    "Dip-Diagnose",
    "Wachstum",
    "Insider",
    "Kapitalrückgabe",
    "Momentum",
    "Bilanzqualität",
]
_AGENTEN_NAMEN_LANG = [
    "Agent 1 – Bewertung",
    "Agent 2 – Drawdown",
    "Agent 3 – Dip-Diagnose",
    "Agent 4 – Wachstum",
    "Agent 5 – Insider/Sentiment",
    "Agent 6 – Kapitalrückgabe",
    "Agent 7 – Momentum",
    "Agent 8 – Bilanzqualität",
]


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def _agent_conviction(ax, befund_dicts: list) -> float:
    befunde = [Befund.from_dict(d) for d in befund_dicts]
    return ax.conviction(befunde)


def _befund_row(b: Befund) -> str:
    if b.bestanden is True:
        icon = "✅"
    elif b.bestanden is False:
        icon = "❌"
    else:
        icon = "⚪"
    wert = f"  `{b.wert}`" if b.wert is not None else ""
    return f"{icon} **{b.label}**{wert}"


def _sterne(conv: float) -> str:
    return "★" * int(conv // 20) + "☆" * (5 - int(conv // 20))


def _conv_farbe(conv: float) -> str:
    if conv >= 70:
        return "🟢"
    if conv >= 50:
        return "🟡"
    return "🔴"


# ── Detail-Ansicht ────────────────────────────────────────────────────────────

def _zeige_agent_block(ax, name_lang: str, befund_dicts: list) -> None:
    befunde = [Befund.from_dict(d) for d in befund_dicts]
    bestanden = sum(1 for b in befunde if b.bestanden is True)
    bewertbar = sum(1 for b in befunde if b.bestanden is not None)
    score = ax.conviction(befunde)
    header = f"{name_lang}  —  {bestanden}/{bewertbar} bestanden  · {score:.0f}%"

    with st.expander(header, expanded=False):
        st.progress(score / 100)
        for b in befunde:
            st.markdown(_befund_row(b))
            if b.details:
                st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ {b.details}")


def _zeige_ergebnis(e: dict) -> None:
    conv = e["conviction"]
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        st.metric("Conviction", f"{conv:.1f} / 100")
    with c2:
        st.markdown(f"<br><span style='font-size:1.6rem'>{_sterne(conv)}</span>",
                    unsafe_allow_html=True)
    with c3:
        st.caption(f"Stand: {e['datum']}  ·  Modell: {e['version']}")

    # Mini-Balken je Agent
    agent_scores = [
        _agent_conviction(ax, e.get(f"agent{i+1}", []))
        for i, ax in enumerate(_AGENTEN)
    ]
    cols = st.columns(8)
    for col, name, score in zip(cols, _AGENTEN_NAMEN, agent_scores):
        with col:
            st.caption(name)
            st.progress(score / 100, text=f"{score:.0f}")

    st.divider()

    left, right = st.columns(2)
    for i, (ax, name_lang) in enumerate(zip(_AGENTEN, _AGENTEN_NAMEN_LANG)):
        col = left if i % 2 == 0 else right
        with col:
            _zeige_agent_block(ax, name_lang, e.get(f"agent{i+1}", []))


# ── App-Layout ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AgentTrade",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📈 AgentTrade")
st.caption("Value Investing Screener · 8 Agenten · Conviction 0–100")
st.divider()

# ── Eingabe ───────────────────────────────────────────────────────────────────

tab_manuell, tab_watchlist = st.tabs(["✏️ Ticker eingeben", "📋 Watchlist"])

tickers: list[str] = []

with tab_manuell:
    tickers_raw = st.text_input(
        "Ticker (kommagetrennt)",
        placeholder="z.B.  AAPL, MSFT, MKL, BRK-B",
        label_visibility="collapsed",
    )
    tickers_manuell = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    if tickers_manuell:
        tickers = tickers_manuell

with tab_watchlist:
    wl_path = Path("watchlist.txt")
    if wl_path.exists():
        wl_tickers = lese_watchlist(wl_path)
        st.caption(f"{len(wl_tickers)} Ticker in watchlist.txt")
        st.code("\n".join(wl_tickers), language=None)
        if st.button("📋 Watchlist screenen", key="wl_btn"):
            tickers = wl_tickers
    else:
        st.info("Keine watchlist.txt gefunden.")

starten = st.button("▶ Screenen", type="primary", disabled=not tickers)

# ── Screening ─────────────────────────────────────────────────────────────────

if starten and tickers:
    ergebnisse: list[dict] = []
    fortschritt = st.progress(0.0, text="Starte …")
    status_box = st.empty()

    for i, ticker in enumerate(tickers):
        status_box.info(f"Analysiere **{ticker}** ({i+1} / {len(tickers)}) …")
        fortschritt.progress(i / len(tickers))
        e = screene(ticker)
        ergebnisse.append(e)

    fortschritt.progress(1.0, text="✅ Fertig")
    status_box.empty()

    # Ranking sortieren
    ergebnisse.sort(key=lambda x: x["conviction"], reverse=True)

    # ── Ranking-Tabelle ──────────────────────────────────────────────────────
    st.subheader("Ranking")

    ranking_rows = []
    for rang, e in enumerate(ergebnisse):
        conv = e["conviction"]
        row = {
            "#": rang + 1,
            "": _conv_farbe(conv),
            "Ticker": e["ticker"],
            "Conviction": conv,
            "Sterne": _sterne(conv),
        }
        for i, name in enumerate(_AGENTEN_NAMEN):
            row[name] = round(_agent_conviction(_AGENTEN[i], e.get(f"agent{i+1}", [])), 0)
        ranking_rows.append(row)

    ranking_df = pd.DataFrame(ranking_rows)

    col_cfg = {
        "Conviction": st.column_config.ProgressColumn(
            "Conviction", min_value=0, max_value=100, format="%.1f"
        )
    }
    for name in _AGENTEN_NAMEN:
        col_cfg[name] = st.column_config.ProgressColumn(
            name, min_value=0, max_value=100, format="%.0f"
        )

    st.dataframe(ranking_df, use_container_width=True, hide_index=True,
                 column_config=col_cfg)

    # ── Detail je Ticker ─────────────────────────────────────────────────────
    st.subheader("Details")
    for e in ergebnisse:
        conv = e["conviction"]
        with st.expander(
            f"{_conv_farbe(conv)} **{e['ticker']}** — {conv:.1f} / 100  {_sterne(conv)}",
            expanded=(len(ergebnisse) == 1),
        ):
            _zeige_ergebnis(e)

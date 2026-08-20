"""AgentTrade – Streamlit Web-Interface."""
from pathlib import Path

import pandas as pd
import streamlit as st

from kern.typen import Befund
from screener import lese_watchlist, screene

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


def _icon(b: Befund) -> str:
    if b.bestanden is None:
        return "⚪"
    return "✅" if b.bestanden else "❌"


def _zeige_agent(name: str, befund_dicts: list) -> None:
    for d in befund_dicts:
        b = Befund.from_dict(d)
        wert = f"  `{b.wert}`" if b.wert is not None else ""
        st.markdown(f"{_icon(b)} **{b.label}**{wert}")
        if b.details:
            st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ {b.details}")


def _zeige_ergebnis(e: dict) -> None:
    conv = e["conviction"]
    sterne = "★" * int(conv // 20) + "☆" * (5 - int(conv // 20))
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("Conviction", f"{conv} / 100")
    with c2:
        st.markdown(f"<h2 style='padding-top:18px'>{sterne}</h2>", unsafe_allow_html=True)
    st.caption(f"Stand: {e['datum']}  ·  Modell: {e['version']}")
    st.divider()
    cols = st.columns(2)
    for i, agent_name in enumerate(_AGENTEN_NAMEN):
        with cols[i % 2]:
            with st.expander(agent_name, expanded=False):
                _zeige_agent(agent_name, e.get(f"agent{i + 1}", []))


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AgentTrade",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📈 AgentTrade")
st.caption("Value Investing Screener – 8 Agenten · Conviction 0–100")

# ── Eingabe ───────────────────────────────────────────────────────────────────

tab_manuell, tab_watchlist = st.tabs(["Ticker eingeben", "Watchlist"])

with tab_manuell:
    tickers_raw = st.text_input(
        "Ticker (kommagetrennt)",
        placeholder="z.B.  AAPL, MSFT, MKL",
        label_visibility="collapsed",
    )
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

with tab_watchlist:
    wl_path = Path("watchlist.txt")
    if wl_path.exists():
        wl_tickers = lese_watchlist(wl_path)
        st.code("\n".join(wl_tickers), language=None)
        if st.button("Watchlist screenen", key="wl_btn"):
            tickers = wl_tickers
    else:
        st.info("Keine watchlist.txt gefunden.")
        wl_tickers = []

starten = st.button("▶ Screenen", type="primary", disabled=not tickers)

# ── Screening ─────────────────────────────────────────────────────────────────

if starten and tickers:
    ergebnisse = []
    fortschritt = st.progress(0, text="Starte …")
    status = st.empty()

    for i, ticker in enumerate(tickers):
        status.info(f"Analysiere **{ticker}** ({i + 1}/{len(tickers)}) …")
        fortschritt.progress(i / len(tickers))
        e = screene(ticker)
        ergebnisse.append(e)

    fortschritt.progress(1.0, text="Fertig.")
    status.empty()

    # Ranking
    ergebnisse.sort(key=lambda x: x["conviction"], reverse=True)
    st.subheader("Ranking")
    ranking_df = pd.DataFrame([
        {
            "#": rang + 1,
            "Ticker": e["ticker"],
            "Conviction": e["conviction"],
            "Sterne": "★" * int(e["conviction"] // 20) + "☆" * (5 - int(e["conviction"] // 20)),
        }
        for rang, e in enumerate(ergebnisse)
    ])
    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Conviction": st.column_config.ProgressColumn(
                "Conviction", min_value=0, max_value=100, format="%.1f"
            )
        },
    )

    # Details
    st.subheader("Details")
    for e in ergebnisse:
        with st.expander(
            f"**{e['ticker']}** — Conviction {e['conviction']} / 100",
            expanded=len(ergebnisse) == 1,
        ):
            _zeige_ergebnis(e)

"""
VCT 2026 Match Predictor — Streamlit Dashboard
Jalankan: streamlit run dashboard.py
"""

import os, json
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="VCT 2026 Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

EVENT_GROUPS = {
    "All": None,
    "Kickoff": ["VCT 2026: Americas Kickoff", "VCT 2026: EMEA Kickoff",
                "VCT 2026: Pacific Kickoff",  "VCT 2026: China Kickoff"],
    "Masters Santiago": ["Valorant Masters Santiago 2026"],
    "Stage 1": ["VCT 2026: Americas Stage 1", "VCT 2026: EMEA Stage 1",
                "VCT 2026: Pacific Stage 1",  "VCT 2026: China Stage 1"],
}

REGION_MAP = {
    "All regions": None,
    "Americas": ["Americas"],
    "EMEA":     ["EMEA"],
    "Pacific":  ["Pacific"],
    "China":    ["China"],
    "International": ["International"],
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_last_updated():
    path = f"{DATA_DIR}/vct2025_matches.csv"
    if not os.path.exists(path):
        return "—"
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).strftime("%d %b %Y, %H:%M")


@st.cache_data
def load_core():
    matches    = pd.read_csv(f"{DATA_DIR}/vct2025_matches.csv",
                             usecols=["event","event_id","match_id","team1","team2",
                                      "score1","score2","date","stage","status","url"])
    team_stats = pd.read_csv(f"{DATA_DIR}/vct2025_team_stats.csv")
    return matches, team_stats


@st.cache_data
def load_player_stats():
    ps_path = f"{DATA_DIR}/vct2025_player_stats.csv"
    pp_path = f"{DATA_DIR}/vct2025_team_player_perf.csv"
    player_stats = pd.read_csv(ps_path) if os.path.exists(ps_path) else pd.DataFrame()
    player_perf  = pd.read_csv(pp_path) if os.path.exists(pp_path) else pd.DataFrame()
    return player_stats, player_perf


@st.cache_resource
def load_model():
    import joblib
    saved = joblib.load(f"{MODEL_DIR}/vct2025_model.pkl")
    return saved["model"], saved["features"]


@st.cache_data
def load_metrics():
    with open(f"{MODEL_DIR}/metrics.json") as f:
        return json.load(f)


def filter_matches(matches, event_filter, region_filter):
    df = matches.copy()
    events = EVENT_GROUPS.get(event_filter)
    if events:
        df = df[df["event"].isin(events)]
    regions = REGION_MAP.get(region_filter)
    if regions:
        # filter berdasarkan nama event yang mengandung region keyword
        region_events = [e for ev_list in [
            EVENT_GROUPS.get(k, []) for k in EVENT_GROUPS
            if k != "All"
        ] for e in ev_list]
        # cari event yang match region
        all_events = pd.read_csv(f"{DATA_DIR}/vct2025_events.csv")
        region_event_names = all_events[all_events["region"].isin(regions)]["name"].tolist()
        if region_event_names:
            df = df[df["event"].isin(region_event_names)]
    return df


def predict(team1, team2, model, feature_cols, team_stats_df, player_perf_df):
    ts = team_stats_df.set_index("team")
    if team1 not in ts.index or team2 not in ts.index:
        return None
    pp = None
    if not player_perf_df.empty and player_perf_df["avg_acs"].notna().any():
        pp = player_perf_df.set_index("team")
    feat = {
        "win_rate_diff":  ts.loc[team1, "win_rate"]      - ts.loc[team2, "win_rate"],
        "form_diff":      ts.loc[team1, "recent_form_5"] - ts.loc[team2, "recent_form_5"],
        "elo_diff":       ts.loc[team1, "elo_final"]     - ts.loc[team2, "elo_final"],
        "h2h_t1_winrate": 0.5,
        "t1_win_rate":    ts.loc[team1, "win_rate"],
        "t2_win_rate":    ts.loc[team2, "win_rate"],
        "t1_recent_form": ts.loc[team1, "recent_form_5"],
        "t2_recent_form": ts.loc[team2, "recent_form_5"],
        "t1_elo":         ts.loc[team1, "elo_final"],
        "t2_elo":         ts.loc[team2, "elo_final"],
    }
    if pp is not None and team1 in pp.index and team2 in pp.index:
        feat["acs_diff"]    = pp.loc[team1, "avg_acs"]    - pp.loc[team2, "avg_acs"]
        feat["rating_diff"] = pp.loc[team1, "avg_rating"] - pp.loc[team2, "avg_rating"]
        feat["t1_avg_acs"]  = pp.loc[team1, "avg_acs"]
        feat["t2_avg_acs"]  = pp.loc[team2, "avg_acs"]
    X = pd.DataFrame([feat]).reindex(columns=feature_cols)
    probs = model.predict_proba(X)[0]
    return {
        "prob1": round(float(probs[1]) * 100, 1),
        "prob2": round(float(probs[0]) * 100, 1),
        "winner": team1 if probs[1] > 0.5 else team2,
        "confidence": round(max(probs) * 100, 1),
    }


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar(matches):
    with st.sidebar:
        st.markdown("## Filters")

        event_filter = st.radio(
            "Event stage",
            options=list(EVENT_GROUPS.keys()),
            index=0,
        )

        st.divider()

        region_filter = st.selectbox(
            "Region",
            options=list(REGION_MAP.keys()),
        )

        st.divider()

        # Data coverage info
        st.markdown("#### Data coverage")
        events_in_data = matches["event"].unique()
        for group, evlist in EVENT_GROUPS.items():
            if group == "All" or evlist is None:
                continue
            count = sum(1 for e in evlist if e in events_in_data)
            total = len(evlist)
            icon  = "✓" if count == total else ("◑" if count > 0 else "○")
            st.caption(f"{icon} {group}: {count}/{total} events")

        st.divider()

        st.markdown("#### Last updated")
        st.caption(get_last_updated())

        st.markdown("#### Source")
        st.caption("[vlr.gg](https://www.vlr.gg) · VCT 2026")

    return event_filter, region_filter


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    matches, team_stats = load_core()
    model, feature_cols = load_model()
    metrics = load_metrics()

    event_filter, region_filter = render_sidebar(matches)
    filtered = filter_matches(matches, event_filter, region_filter)

    # Recompute team stats dari filtered matches jika filter aktif
    if event_filter != "All" or region_filter != "All regions":
        filtered_teams = set(filtered["team1"]) | set(filtered["team2"])
        ts_filtered = team_stats[team_stats["team"].isin(filtered_teams)].copy()
    else:
        ts_filtered = team_stats.copy()

    team_list = sorted(ts_filtered["team"].tolist())

    # ── HEADER ────────────────────────────────
    col_h, col_badge = st.columns([4, 1])
    with col_h:
        st.markdown("# 🎯 VCT 2026 Match Predictor")
        filter_label = event_filter if event_filter != "All" else "All stages"
        region_label = region_filter if region_filter != "All regions" else "All regions"
        st.caption(f"Showing: **{filter_label}** · **{region_label}** · Last updated: {get_last_updated()}")
    with col_badge:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='text-align:right'>"
            f"<span style='background:#E1F5EE;color:#085041;padding:4px 10px;"
            f"border-radius:6px;font-size:12px;font-weight:500'>"
            f"{len(filtered)} matches · {len(ts_filtered)} teams</span></div>",
            unsafe_allow_html=True
        )

    st.divider()

    # ── METRIC CARDS ─────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Matches (filtered)", len(filtered))
    c2.metric("Teams",              len(ts_filtered))
    c3.metric("Events",             filtered["event"].nunique())
    c4.metric("CV Accuracy",        f"{metrics['cv_accuracy_mean']*100:.1f}%")
    c5.metric("ROC-AUC",            f"{metrics['cv_roc_auc_mean']:.3f}")

    st.divider()

    # ── TABS ─────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚔️ Prediksi Match",
        "🏆 Elo Ranking",
        "📊 Statistik Tim",
        "📅 Match History",
        "🗺️ Analisis Map",
    ])

    # ══════════════════════════════════════════
    # TAB 1 — PREDIKSI
    # ══════════════════════════════════════════
    with tab1:
        _, player_perf = load_player_stats()
        st.subheader("Prediksi Hasil Match")
        st.caption("Berdasarkan Elo rating, win rate, recent form, dan head-to-head dari data yang difilter")

        if len(team_list) < 2:
            st.warning("Tidak cukup tim di filter ini. Coba perluas filter.")
        else:
            col_l, col_r = st.columns(2)
            with col_l:
                default_t1 = "NRG" if "NRG" in team_list else team_list[0]
                team1 = st.selectbox("Tim 1", team_list,
                                     index=team_list.index(default_t1))
            with col_r:
                remaining = [t for t in team_list if t != team1]
                default_t2 = "G2 Esports" if "G2 Esports" in remaining else remaining[0]
                team2 = st.selectbox("Tim 2", remaining,
                                     index=remaining.index(default_t2) if default_t2 in remaining else 0)

            if st.button("🔮 Prediksi", use_container_width=True, type="primary"):
                result = predict(team1, team2, model, feature_cols, ts_filtered, player_perf)
                if result:
                    st.markdown("---")
                    r1, r2 = st.columns(2)
                    with r1:
                        badge = "🏆 " if result["winner"] == team1 else ""
                        st.metric(f"{badge}{team1}", f"{result['prob1']}%",
                                  delta=f"{result['prob1']-50:+.1f}% vs baseline")
                    with r2:
                        badge = "🏆 " if result["winner"] == team2 else ""
                        st.metric(f"{badge}{team2}", f"{result['prob2']}%",
                                  delta=f"{result['prob2']-50:+.1f}% vs baseline")

                    fig = go.Figure(go.Bar(
                        x=[result["prob1"], result["prob2"]],
                        y=[team1, team2],
                        orientation="h",
                        marker_color=["#E84057" if result["winner"] == t else "#636EFA"
                                      for t in [team1, team2]],
                        text=[f"{result['prob1']}%", f"{result['prob2']}%"],
                        textposition="inside",
                        textfont=dict(size=14, color="white"),
                    ))
                    fig.update_layout(
                        xaxis=dict(range=[0,100], title="Win Probability (%)"),
                        yaxis=dict(title=""),
                        height=160, margin=dict(l=0,r=0,t=8,b=0),
                        showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(f"**Predicted winner: {result['winner']}** — {result['confidence']}% confidence")

                    with st.expander("📋 Detail faktor prediksi"):
                        ts_idx = ts_filtered.set_index("team")
                        st.dataframe(pd.DataFrame({
                            "Faktor":  ["Elo Rating", "Win Rate", "Recent Form (5)", "Data source"],
                            team1: [
                                f"{ts_idx.loc[team1,'elo_final']:.0f}",
                                f"{ts_idx.loc[team1,'win_rate']*100:.1f}%",
                                f"{ts_idx.loc[team1,'recent_form_5']*100:.0f}%",
                                event_filter,
                            ],
                            team2: [
                                f"{ts_idx.loc[team2,'elo_final']:.0f}",
                                f"{ts_idx.loc[team2,'win_rate']*100:.1f}%",
                                f"{ts_idx.loc[team2,'recent_form_5']*100:.0f}%",
                                event_filter,
                            ],
                        }), use_container_width=True, hide_index=True)

            # Quick matchups Masters London 2026
            st.markdown("---")
            st.markdown("**Masters London 2026 — Quick pick'em:**")
            quick = [("G2 Esports","NRG"), ("Paper Rex","Team Heretics"),
                     ("EDward Gaming","Team Vitality"), ("Leviatán","FULL SENSE")]
            qcols = st.columns(len(quick))
            for i, (t1, t2) in enumerate(quick):
                if t1 in team_list and t2 in team_list:
                    r = predict(t1, t2, model, feature_cols, ts_filtered, player_perf)
                    if r:
                        with qcols[i]:
                            st.markdown(f"**{t1}** vs **{t2}**")
                            st.progress(r["prob1"] / 100)
                            st.caption(f"{t1} {r['prob1']}% · {t2} {r['prob2']}%")

    # ══════════════════════════════════════════
    # TAB 2 — ELO RANKING
    # ══════════════════════════════════════════
    with tab2:
        st.subheader("Elo Rating Ranking")
        st.caption(f"Dihitung temporal dari data: {event_filter}")

        col_f, col_r2, _ = st.columns([1, 1, 2])
        with col_f:
            top_n = st.slider("Top N tim", 5, min(40, len(ts_filtered)), 15)
        with col_r2:
            sort_by = st.selectbox("Urutkan", ["Elo Rating", "Win Rate", "Recent Form"])

        sort_col = {"Elo Rating": "elo_final",
                    "Win Rate":   "win_rate",
                    "Recent Form": "recent_form_5"}[sort_by]

        top = ts_filtered.nlargest(top_n, sort_col)

        fig = go.Figure(go.Bar(
            x=top[sort_col],
            y=top["team"],
            orientation="h",
            marker=dict(color=top[sort_col], colorscale="Reds", showscale=False),
            text=top[sort_col].apply(
                lambda v: f"{v:.0f}" if sort_by == "Elo Rating" else f"{v*100:.1f}%"
            ),
            textposition="outside",
        ))
        x_range = [top[sort_col].min() * 0.97, top[sort_col].max() * 1.06] \
            if sort_by == "Elo Rating" else [0, 1.1]
        fig.update_layout(
            xaxis=dict(title=sort_by, range=x_range),
            yaxis=dict(title="", autorange="reversed"),
            height=max(300, top_n * 28),
            margin=dict(l=0, r=70, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        disp = ts_filtered[["team","elo_final","total_matches","wins","losses",
                             "win_rate","recent_form_5"]].copy()
        disp.columns = ["Tim","Elo","Matches","Wins","Losses","Win Rate","Form (5)"]
        disp["Win Rate"] = (disp["Win Rate"] * 100).round(1).astype(str) + "%"
        disp["Form (5)"] = (disp["Form (5)"] * 100).round(0).astype(int).astype(str) + "%"
        disp["Elo"]      = disp["Elo"].round(0).astype(int)
        st.dataframe(disp.sort_values("Elo", ascending=False).reset_index(drop=True),
                     use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════
    # TAB 3 — STATISTIK TIM
    # ══════════════════════════════════════════
    with tab3:
        player_stats, player_perf = load_player_stats()
        st.subheader("Statistik Tim")

        if not team_list:
            st.warning("Tidak ada tim di filter ini.")
        else:
            selected = st.selectbox("Pilih tim", team_list, key="team_stat_sel")
            ts_idx   = ts_filtered.set_index("team")

            if selected in ts_idx.index:
                row = ts_idx.loc[selected]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Elo",         f"{row['elo_final']:.0f}")
                m2.metric("Win Rate",    f"{row['win_rate']*100:.1f}%")
                m3.metric("Recent Form", f"{row['recent_form_5']*100:.0f}%")
                m4.metric("Matches",     int(row["total_matches"]))

            st.markdown("---")
            st.markdown(f"**Match history — {selected}** *(filter: {event_filter})*")

            tm = filtered[(filtered["team1"] == selected) | (filtered["team2"] == selected)].copy()
            tm["Hasil"]  = tm.apply(
                lambda r: "Win" if (
                    (r["team1"] == selected and r["score1"] > r["score2"]) or
                    (r["team2"] == selected and r["score2"] > r["score1"])
                ) else "Loss", axis=1
            )
            tm["Lawan"] = tm.apply(
                lambda r: r["team2"] if r["team1"] == selected else r["team1"], axis=1
            )
            tm["Skor"]  = tm.apply(
                lambda r: f"{int(r['score1'])}–{int(r['score2'])}"
                if r["team1"] == selected else f"{int(r['score2'])}–{int(r['score1'])}", axis=1
            )
            show = tm[["event","Lawan","Skor","Hasil"]].copy()
            show.columns = ["Event","Lawan","Skor","Hasil"]
            st.dataframe(
                show.style.map(
                    lambda v: "color: #27ae60" if v == "Win" else "color: #e74c3c",
                    subset=["Hasil"]
                ),
                use_container_width=True, hide_index=True,
            )

            if not player_perf.empty and player_perf["avg_acs"].notna().any():
                pp_idx = player_perf.set_index("team")
                if selected in pp_idx.index:
                    st.markdown("---")
                    st.markdown(f"**Avg player performance — {selected}**")
                    p = pp_idx.loc[selected]
                    pc1, pc2, pc3, pc4 = st.columns(4)
                    pc1.metric("Avg ACS",    f"{p['avg_acs']:.0f}")
                    pc2.metric("Avg Rating", f"{p['avg_rating']:.2f}")
                    pc3.metric("Avg ADR",    f"{p['avg_adr']:.1f}")
                    pc4.metric("Avg K/D",    f"{p['avg_kills']/max(p['avg_deaths'],1):.2f}")

    # ══════════════════════════════════════════
    # TAB 4 — MATCH HISTORY
    # ══════════════════════════════════════════
    with tab4:
        st.subheader("Match History")
        st.caption(f"Filter aktif: {event_filter} · {region_filter}")

        col_s, col_r3 = st.columns([2, 1])
        with col_s:
            search = st.text_input("Cari tim", placeholder="Ketik nama tim...")
        with col_r3:
            result_filter = st.selectbox("Hasil", ["Semua", "Completed", "Upcoming"])

        disp = filtered.copy()

        if search:
            mask = (disp["team1"].str.contains(search, case=False, na=False) |
                    disp["team2"].str.contains(search, case=False, na=False))
            disp = disp[mask]

        if result_filter == "Completed":
            disp = disp[disp["score1"].notna() & (disp["score1"].astype(str) != "")]
        elif result_filter == "Upcoming":
            disp = disp[disp["score1"].isna() | (disp["score1"].astype(str) == "")]

        st.markdown(f"**{len(disp)} matches ditemukan**")

        show = disp[["event","stage","team1","score1","score2","team2","status"]].copy()
        show.columns = ["Event","Stage","Team 1","Score 1","Score 2","Team 2","Status"]
        st.dataframe(show.reset_index(drop=True),
                     use_container_width=True, hide_index=True)

        # Win rate per event
        st.markdown("---")
        st.markdown("**Win rate per event (filtered)**")
        ev_stats = []
        for ev_name, grp in filtered.groupby("event"):
            completed = grp[grp["score1"].notna() &
                            (grp["score1"].astype(str).str.strip() != "")]
            ev_stats.append({
                "Event": ev_name,
                "Matches": len(grp),
                "Completed": len(completed),
            })
        if ev_stats:
            st.dataframe(pd.DataFrame(ev_stats), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════
    # TAB 5 — MAP ANALYSIS
    # ══════════════════════════════════════════
    with tab5:
        player_stats, player_perf = load_player_stats()
        st.subheader("Analisis Map")

        ps = player_stats.copy()
        # Filter player stats sesuai event filter
        if event_filter != "All":
            events_sel = EVENT_GROUPS.get(event_filter, [])
            if events_sel:
                ps = ps[ps["event"].isin(events_sel)]

        ps["map_clean"] = ps["map"].str.extract(r'^([A-Za-z]+)', expand=False)
        ps = ps[ps["map_clean"].notna() & (ps["map_clean"] != "")]
        map_counts = ps.groupby("map_clean")["match_id"].nunique().sort_values(ascending=False)
        map_counts = map_counts[map_counts > 2]

        if map_counts.empty:
            st.info("Tidak cukup data map di filter ini.")
        else:
            col_m, _ = st.columns([2, 2])
            with col_m:
                sel_map = st.selectbox("Pilih map", map_counts.index.tolist())

            map_data = ps[ps["map_clean"] == sel_map].copy()

            if not map_data.empty and map_data["acs"].notna().any():
                st.markdown(f"**Top performers di {sel_map}** (min. 2 maps · filter: {event_filter})")
                pm = (
                    map_data.groupby("player")
                    .agg(avg_acs=("acs","mean"), avg_rating=("rating","mean"),
                         avg_kills=("kills","mean"), maps_played=("match_id","nunique"))
                    .reset_index().query("maps_played >= 2")
                    .sort_values("avg_acs", ascending=False).head(15)
                )
                if not pm.empty:
                    fig = px.bar(
                        pm, x="avg_acs", y="player", orientation="h",
                        color="avg_rating", color_continuous_scale="Reds",
                        labels={"avg_acs":"Avg ACS","player":"Player","avg_rating":"Avg Rating"},
                        text=pm["avg_acs"].round(0).astype(int),
                    )
                    fig.update_layout(
                        height=max(300, len(pm)*30),
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=0,r=0,t=10,b=0),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Data player stats belum tersedia untuk map ini.")

    # ── FOOTER ───────────────────────────────
    st.divider()
    st.caption(
        f"Data: [vlr.gg](https://www.vlr.gg) · "
        f"VCT 2026 (Kickoff + Masters Santiago + Stage 1) · "
        f"Model: Ensemble (LR + GBM) · "
        f"Fitur: Temporal Elo, Win Rate, Recent Form, H2H, ACS · "
        f"Last updated: {get_last_updated()}"
    )


if __name__ == "__main__":
    main()

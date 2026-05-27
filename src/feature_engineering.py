"""
Feature Engineering — VCT 2025 (v2 - temporal Elo, no leakage)
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_data():
    matches = pd.read_csv(f"{DATA_DIR}/vct2025_matches.csv")
    player_stats = pd.read_csv(f"{DATA_DIR}/vct2025_player_stats.csv")

    matches = matches[matches["score1"].notna() & matches["score2"].notna()].copy()
    matches["score1"] = pd.to_numeric(matches["score1"], errors="coerce")
    matches["score2"] = pd.to_numeric(matches["score2"], errors="coerce")
    matches = matches.dropna(subset=["score1", "score2"])
    matches["winner"] = matches.apply(
        lambda r: r["team1"] if r["score1"] > r["score2"] else r["team2"], axis=1
    )
    # Urutkan berdasarkan index (proxy urutan kronologis dari scraper)
    matches = matches.reset_index(drop=True)
    return matches, player_stats


# ─────────────────────────────────────────────
# ELO TEMPORAL (no leakage)
# ─────────────────────────────────────────────

def compute_elo_temporal(matches: pd.DataFrame, k: int = 32, initial: int = 1500) -> pd.DataFrame:
    """
    Hitung Elo SEBELUM setiap match dimainkan.
    Return: DataFrame dengan kolom t1_elo, t2_elo ditambahkan ke matches.
    Ini menghindari data leakage — model hanya tahu Elo historis saat prediksi.
    """
    elo = {}
    t1_elos, t2_elos = [], []

    for _, row in matches.iterrows():
        t1, t2 = row["team1"], row["team2"]
        elo.setdefault(t1, initial)
        elo.setdefault(t2, initial)

        # Catat Elo SEBELUM match ini
        t1_elos.append(elo[t1])
        t2_elos.append(elo[t2])

        # Update Elo SETELAH match
        e1 = 1 / (1 + 10 ** ((elo[t2] - elo[t1]) / 400))
        e2 = 1 - e1
        s1 = 1.0 if row["winner"] == t1 else 0.0
        s2 = 1.0 - s1
        elo[t1] += k * (s1 - e1)
        elo[t2] += k * (s2 - e2)

    matches = matches.copy()
    matches["t1_elo"] = t1_elos
    matches["t2_elo"] = t2_elos
    matches["elo_diff"] = matches["t1_elo"] - matches["t2_elo"]

    # Elo final (untuk display)
    print(f"  Top 10 Elo (final):")
    top = sorted(elo.items(), key=lambda x: x[1], reverse=True)[:10]
    for team, score in top:
        print(f"    {team:<30} {score:.0f}")

    return matches, elo


# ─────────────────────────────────────────────
# WIN RATE & RECENT FORM (temporal)
# ─────────────────────────────────────────────

def compute_rolling_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung win_rate dan recent_form SEBELUM setiap match (temporal).
    Menghindari leakage dengan hanya memakai match sebelum index saat ini.
    """
    win_rates_t1, win_rates_t2 = [], []
    form_t1, form_t2 = [], []

    for i, row in matches.iterrows():
        t1, t2 = row["team1"], row["team2"]

        # Semua match SEBELUM match ini
        past = matches.loc[:i-1] if i > 0 else matches.loc[[]]

        def get_stats(team, past_df):
            tm = past_df[(past_df["team1"] == team) | (past_df["team2"] == team)]
            if len(tm) == 0:
                return 0.5, 0.5  # prior netral kalau belum ada history
            wins = (tm["winner"] == team).sum()
            wr = wins / len(tm)
            recent = tm.tail(5)
            rf = (recent["winner"] == team).sum() / len(recent)
            return round(wr, 4), round(rf, 4)

        wr1, rf1 = get_stats(t1, past)
        wr2, rf2 = get_stats(t2, past)

        win_rates_t1.append(wr1)
        win_rates_t2.append(wr2)
        form_t1.append(rf1)
        form_t2.append(rf2)

    matches = matches.copy()
    matches["t1_win_rate"]    = win_rates_t1
    matches["t2_win_rate"]    = win_rates_t2
    matches["win_rate_diff"]  = matches["t1_win_rate"] - matches["t2_win_rate"]
    matches["t1_recent_form"] = form_t1
    matches["t2_recent_form"] = form_t2
    matches["form_diff"]      = matches["t1_recent_form"] - matches["t2_recent_form"]
    return matches


# ─────────────────────────────────────────────
# HEAD-TO-HEAD (temporal)
# ─────────────────────────────────────────────

def compute_h2h_temporal(matches: pd.DataFrame) -> pd.DataFrame:
    """H2H win rate dihitung dari match sebelum match saat ini."""
    h2h_winrates = []

    for i, row in matches.iterrows():
        t1, t2 = row["team1"], row["team2"]
        past = matches.loc[:i-1] if i > 0 else matches.loc[[]]

        h2h = past[
            ((past["team1"] == t1) & (past["team2"] == t2)) |
            ((past["team1"] == t2) & (past["team2"] == t1))
        ]

        if len(h2h) == 0:
            h2h_winrates.append(0.5)  # prior netral
        else:
            a_wins = (h2h["winner"] == t1).sum()
            h2h_winrates.append(round(a_wins / len(h2h), 4))

    matches = matches.copy()
    matches["h2h_t1_winrate"] = h2h_winrates
    return matches


# ─────────────────────────────────────────────
# TEAM STATS (untuk display & prediksi baru)
# ─────────────────────────────────────────────

def compute_team_stats_final(matches: pd.DataFrame, elo: dict) -> pd.DataFrame:
    teams = set(matches["team1"]) | set(matches["team2"])
    records = []
    for team in teams:
        tm = matches[(matches["team1"] == team) | (matches["team2"] == team)]
        wins  = (tm["winner"] == team).sum()
        total = len(tm)
        recent = tm.tail(5)
        records.append({
            "team":          team,
            "total_matches": total,
            "wins":          int(wins),
            "losses":        total - int(wins),
            "win_rate":      round(wins/total, 4) if total > 0 else 0.5,
            "recent_form_5": round((recent["winner"] == team).sum() / len(recent), 4) if len(recent) > 0 else 0.5,
            "elo_final":     round(elo.get(team, 1500), 1),
        })
    df = pd.DataFrame(records).sort_values("elo_final", ascending=False)
    df.to_csv(f"{DATA_DIR}/vct2025_team_stats.csv", index=False)
    return df


# ─────────────────────────────────────────────
# PLAYER PERFORMANCE PER TIM
# ─────────────────────────────────────────────

def compute_team_player_stats(player_stats: pd.DataFrame) -> pd.DataFrame:
    if player_stats.empty:
        return pd.DataFrame()

    ps = player_stats.copy()

    # Drop baris yang semua stats-nya NaN
    stat_cols = ["acs", "rating", "adr", "kills", "deaths"]
    ps = ps.dropna(subset=stat_cols, how="all")
    if ps.empty:
        print("  [!] Player stats semua NaN — skip")
        return pd.DataFrame()

    ps["row_num"] = ps.groupby(["match_id", "map"]).cumcount()
    ps["player_team"] = ps.apply(
        lambda r: r["team1"] if r["row_num"] < 5 else r["team2"], axis=1
    )

    agg = ps.groupby("player_team").agg(
        avg_acs=("acs", "mean"),
        avg_rating=("rating", "mean"),
        avg_adr=("adr", "mean"),
        avg_kills=("kills", "mean"),
        avg_deaths=("deaths", "mean"),
    ).round(3).reset_index()
    agg.rename(columns={"player_team": "team"}, inplace=True)

    non_null = agg["avg_acs"].notna().sum()
    print(f"  -> {non_null}/{len(agg)} tim punya player stats")
    return agg


# ─────────────────────────────────────────────
# BUILD FEATURE MATRIX
# ─────────────────────────────────────────────

def build_feature_matrix(matches: pd.DataFrame, player_perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pp = player_perf.set_index("team") if not player_perf.empty else pd.DataFrame()

    for _, row in matches.iterrows():
        t1, t2 = row["team1"], row["team2"]

        feat = {
            "match_id":       row.get("match_id", ""),
            "event":          row.get("event", ""),
            "team1":          t1,
            "team2":          t2,
            "t1_win_rate":    row["t1_win_rate"],
            "t2_win_rate":    row["t2_win_rate"],
            "win_rate_diff":  row["win_rate_diff"],
            "t1_recent_form": row["t1_recent_form"],
            "t2_recent_form": row["t2_recent_form"],
            "form_diff":      row["form_diff"],
            "t1_elo":         row["t1_elo"],
            "t2_elo":         row["t2_elo"],
            "elo_diff":       row["elo_diff"],
            "h2h_t1_winrate": row["h2h_t1_winrate"],
        }

        if not pp.empty and t1 in pp.index and t2 in pp.index:
            feat["acs_diff"]    = pp.loc[t1, "avg_acs"]    - pp.loc[t2, "avg_acs"]
            feat["rating_diff"] = pp.loc[t1, "avg_rating"] - pp.loc[t2, "avg_rating"]
            feat["t1_avg_acs"]  = pp.loc[t1, "avg_acs"]
            feat["t2_avg_acs"]  = pp.loc[t2, "avg_acs"]

        feat["label"] = 1 if row["winner"] == t1 else 0
        rows.append(feat)

    df = pd.DataFrame(rows)
    df.to_csv(f"{DATA_DIR}/vct2025_features.csv", index=False)
    print(f"  -> Feature matrix: {df.shape} -> data/vct2025_features.csv")
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  VCT 2025 Feature Engineering (v2 - temporal)")
    print("=" * 55)

    matches, player_stats = load_data()
    print(f"\nLoaded: {len(matches)} matches, {len(player_stats)} player stat rows")

    print("\n[1/4] Temporal Elo ratings...")
    matches, elo = compute_elo_temporal(matches)

    print("\n[2/4] Rolling win rate & recent form...")
    matches = compute_rolling_stats(matches)
    print(f"  -> Done ({len(matches)} matches)")

    print("\n[3/4] Temporal H2H...")
    matches = compute_h2h_temporal(matches)
    print(f"  -> Done")

    print("\n[4/4] Player performance per tim...")
    player_perf = compute_team_player_stats(player_stats)
    if not player_perf.empty:
        player_perf.to_csv(f"{DATA_DIR}/vct2025_team_player_perf.csv", index=False)
        print(player_perf.head(5).to_string(index=False))

    print("\n[5/5] Building feature matrix...")
    features_df = build_feature_matrix(matches, player_perf if not player_perf.empty else pd.DataFrame())

    # Simpan team stats final
    team_stats = compute_team_stats_final(matches, elo)
    print(f"\n  Top 10 tim (by Elo):")
    print(team_stats[["team","total_matches","wins","win_rate","elo_final"]].head(10).to_string(index=False))

    print("\n✅ Feature engineering selesai!")

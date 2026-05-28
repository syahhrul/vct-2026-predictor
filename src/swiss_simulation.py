"""
VCT Masters London 2026 — Swiss Stage Monte Carlo Simulation
Format: GSL Swiss, Bo3, advance 2W, eliminate 2L, top 4 lanjut playoff
Jalankan: python src/swiss_simulation.py
"""

import os, json, pickle
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm
import joblib

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ─────────────────────────────────────────────
# 8 TIM SWISS STAGE MASTERS LONDON 2026
# (4 tim lainnya langsung ke playoff sebagai seed 1 tiap region)
# ─────────────────────────────────────────────
SWISS_TEAMS = [
    "Leviatán",       # AMER #2
    "NRG",            # AMER #3
    "Xi Lai Gaming",  # CN #2
    "Dragon Ranger Gaming",  # CN #3
    "Team Vitality",  # EMEA #2
    "FUT Esports",    # EMEA #3
    "FULL SENSE",     # PAC #2
    "Global Esports", # PAC #3
]

# Direct seeds ke playoff (dari VLR.gg resmi)
# Americas #1 masih TBD — placeholder pakai G2 (juara Americas Stage 1 terkuat)
PLAYOFF_DIRECT = [
    "G2 Esports",    # AMER #1 (TBD — placeholder)
    "EDward Gaming", # CN #1
    "Team Heretics", # EMEA #1
    "Paper Rex",     # PAC #1
]

# Format playoff resmi dari VLR.gg:
# UB QF (x4) → UB SF (x2) → UB Final → Grand Final
# LR1 (x2) → LR2 (x2) → LR3 (x1) → Lower Final → Grand Final
# Total 8 tim playoff (4 direct + 4 Swiss)

# Playoff bracket seeds setelah Swiss
# 4 tim dari Swiss + 4 direct seeds = 8 tim playoff
PLAYOFF_TEAMS_ORDER = PLAYOFF_DIRECT  # akan ditambah 4 swiss qualifiers


def get_win_probability(team1: str, team2: str, model, feature_cols,
                        team_stats: pd.DataFrame) -> float:
    """
    Hitung win probability team1 vs team2.
    Kalau tim tidak ada di model, pakai Elo-based estimate.
    """
    ts = team_stats.set_index("team")

    # Kalau kedua tim ada di model
    if team1 in ts.index and team2 in ts.index:
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
        X     = pd.DataFrame([feat]).reindex(columns=feature_cols)
        probs = model.predict_proba(X)[0]
        return float(probs[1])

    # Fallback: Elo-based estimate kalau tim tidak ada
    elo_map = {
        # VCT 2026 Stage 1 approximate Elo
        "G2 Esports":           1750,
        "Paper Rex":            1680,
        "Team Heretics":        1660,
        "EDward Gaming":        1640,
        "NRG":                  1620,
        "Team Vitality":        1600,
        "Leviatán":             1590,
        "FULL SENSE":           1520,
        "FUT Esports":          1500,
        "Xi Lai Gaming":        1480,
        "Dragon Ranger Gaming": 1460,
        "Global Esports":       1440,
    }
    e1 = elo_map.get(team1, 1500)
    e2 = elo_map.get(team2, 1500)
    return 1 / (1 + 10 ** ((e2 - e1) / 400))


# ─────────────────────────────────────────────
# SWISS STAGE SIMULATION
# Format GSL: ronde 1 random/seeded, ronde 2-3 same-record matchup
# Advance: 2W, Eliminate: 2L, max 3 ronde
# ─────────────────────────────────────────────

def simulate_match(team1: str, team2: str, win_probs: dict) -> str:
    """Simulate satu match, return winner."""
    key   = (team1, team2)
    rkey  = (team2, team1)
    if key in win_probs:
        p = win_probs[key]
    elif rkey in win_probs:
        p = 1 - win_probs[rkey]
    else:
        p = 0.5
    return team1 if np.random.random() < p else team2


def run_swiss_simulation(win_probs: dict, n_simulations: int = 10000) -> dict:
    """
    Jalankan N simulasi Swiss stage.
    Return: dict {team: advance_count}
    """
    advance_counts = defaultdict(int)
    round_records  = defaultdict(lambda: defaultdict(int))  # team -> ronde advance

    for _ in range(n_simulations):
        # State: {team: (wins, losses)}
        records = {t: [0, 0] for t in SWISS_TEAMS}
        active  = set(SWISS_TEAMS)
        advanced = []
        eliminated = []

        # Ronde 1 — random pairing (no intra-regional constraint untuk simplifikasi)
        teams_list = list(active)
        np.random.shuffle(teams_list)
        matchups_r1 = [(teams_list[i], teams_list[i+1])
                       for i in range(0, len(teams_list), 2)]

        for t1, t2 in matchups_r1:
            winner = simulate_match(t1, t2, win_probs)
            loser  = t2 if winner == t1 else t1
            records[winner][0] += 1
            records[loser][1]  += 1

        # Ronde 2 & 3 — same-record matchup
        for ronde in range(2, 4):
            # Kelompokkan berdasarkan record
            groups = defaultdict(list)
            for t in active:
                if records[t][0] < 2 and records[t][1] < 2:
                    key = (records[t][0], records[t][1])
                    groups[key].append(t)

            # Match dalam grup yang sama record
            for rec, group in groups.items():
                np.random.shuffle(group)
                # Kalau jumlah ganjil, satu tim dapat bye (match vs tim record berbeda)
                if len(group) % 2 == 1:
                    # Cari tim dari grup terdekat untuk jadi pasangan
                    pass  # simplifikasi: skip (jarang terjadi di 8 tim)
                for i in range(0, len(group) - 1, 2):
                    t1, t2 = group[i], group[i+1]
                    winner = simulate_match(t1, t2, win_probs)
                    loser  = t2 if winner == t1 else t1
                    records[winner][0] += 1
                    records[loser][1]  += 1

            # Cek advance/eliminate setelah ronde ini
            for t in list(active):
                w, l = records[t]
                if w == 2:
                    advanced.append(t)
                    active.discard(t)
                    advance_counts[t] += 1
                elif l == 2:
                    eliminated.append(t)
                    active.discard(t)

        # Sisa tim setelah 3 ronde (harusnya sudah semua resolved)
        for t in active:
            w, l = records[t]
            if w > l:
                advanced.append(t)
                advance_counts[t] += 1

    return advance_counts, n_simulations


# ─────────────────────────────────────────────
# PLAYOFF SIMULATION (double elimination)
# ─────────────────────────────────────────────

def simulate_playoff_once(playoff_8: list, win_probs: dict) -> str:
    """
    Simulate satu skenario double elimination 8 tim sesuai format VLR.gg resmi:
    UB QF (4 match) → UB SF (2 match) → UB Final
    LR1 (2 match) → LR2 (2 match) → LR3 (1 match) → Lower Final
    Grand Final: UB Final winner vs Lower Final winner
    Seeding: 4 direct seeds (1-4) vs 4 Swiss qualifiers (5-8)
    """
    seeds = playoff_8[:4]   # direct seeds — 1,2,3,4
    quals = playoff_8[4:]   # swiss qualifiers — 5,6,7,8
    np.random.shuffle(quals)  # urutan Swiss qualifiers random

    # UB Quarterfinals: 1v8, 2v7, 3v6, 4v5
    ub_qf_matchups = [
        (seeds[0], quals[3]),  # 1 vs 8
        (seeds[1], quals[2]),  # 2 vs 7
        (seeds[2], quals[1]),  # 3 vs 6
        (seeds[3], quals[0]),  # 4 vs 5
    ]
    ub_qf_winners = []
    lr1_teams     = []
    for t1, t2 in ub_qf_matchups:
        w = simulate_match(t1, t2, win_probs)
        l = t2 if w == t1 else t1
        ub_qf_winners.append(w)
        lr1_teams.append(l)

    # UB Semifinals: QF winner 1v2, QF winner 3v4
    ub_sf_matchups = [
        (ub_qf_winners[0], ub_qf_winners[1]),
        (ub_qf_winners[2], ub_qf_winners[3]),
    ]
    ub_sf_winners = []
    ub_sf_losers  = []
    for t1, t2 in ub_sf_matchups:
        w = simulate_match(t1, t2, win_probs)
        l = t2 if w == t1 else t1
        ub_sf_winners.append(w)
        ub_sf_losers.append(l)

    # UB Final
    ub_final_w = simulate_match(ub_sf_winners[0], ub_sf_winners[1], win_probs)
    ub_final_l = ub_sf_winners[1] if ub_final_w == ub_sf_winners[0] else ub_sf_winners[0]

    # LR1: 2 match dari UB QF losers
    np.random.shuffle(lr1_teams)
    lr1_winners = []
    for i in range(0, 4, 2):
        w = simulate_match(lr1_teams[i], lr1_teams[i+1], win_probs)
        lr1_winners.append(w)

    # LR2: LR1 winners vs UB SF losers
    lr2_winners = []
    for i in range(2):
        w = simulate_match(ub_sf_losers[i], lr1_winners[i], win_probs)
        lr2_winners.append(w)

    # LR3: LR2 winner 1 vs LR2 winner 2
    lr3_w = simulate_match(lr2_winners[0], lr2_winners[1], win_probs)

    # Lower Final: LR3 winner vs UB Final loser
    lower_final_w = simulate_match(ub_final_l, lr3_w, win_probs)

    # Grand Final
    champion = simulate_match(ub_final_w, lower_final_w, win_probs)
    return champion


def run_full_simulation(win_probs: dict, n: int = 10000) -> dict:
    """
    Jalankan simulasi penuh: Swiss → Playoff.
    Return probabilitas champion tiap tim.
    """
    all_teams      = SWISS_TEAMS + PLAYOFF_DIRECT
    champion_count = defaultdict(int)
    finalist_count = defaultdict(int)
    top4_count     = defaultdict(int)

    for _ in tqdm(range(n), desc="Simulating"):
        # Swiss stage
        swiss_advanced = []
        records = {t: [0, 0] for t in SWISS_TEAMS}
        active  = set(SWISS_TEAMS)

        teams_list = list(active)
        np.random.shuffle(teams_list)
        matchups_r1 = [(teams_list[i], teams_list[i+1])
                       for i in range(0, len(teams_list), 2)]

        for t1, t2 in matchups_r1:
            w = simulate_match(t1, t2, win_probs)
            l = t2 if w == t1 else t1
            records[w][0] += 1
            records[l][1]  += 1

        for ronde in range(2, 4):
            groups = defaultdict(list)
            for t in list(active):
                if records[t][0] < 2 and records[t][1] < 2:
                    groups[(records[t][0], records[t][1])].append(t)

            for rec, group in groups.items():
                np.random.shuffle(group)
                for i in range(0, len(group) - 1, 2):
                    t1, t2 = group[i], group[i+1]
                    w = simulate_match(t1, t2, win_probs)
                    l = t2 if w == t1 else t1
                    records[w][0] += 1
                    records[l][1]  += 1

            for t in list(active):
                w, l = records[t]
                if w == 2:
                    swiss_advanced.append(t)
                    active.discard(t)
                elif l == 2:
                    active.discard(t)

        for t in active:
            if records[t][0] > records[t][1]:
                swiss_advanced.append(t)

        # Ambil 4 tim Swiss (kalau lebih karena simulasi, acak)
        swiss_advanced = swiss_advanced[:4]
        if len(swiss_advanced) < 4:
            remaining = [t for t in SWISS_TEAMS if t not in swiss_advanced]
            swiss_advanced += remaining[:4 - len(swiss_advanced)]

        # Playoff 8 tim
        playoff_8 = PLAYOFF_DIRECT + swiss_advanced

        # Simulate playoff
        champ = simulate_playoff_once(playoff_8, win_probs)
        champion_count[champ] += 1

        # Top 4 heuristic (semua playoff tim)
        for t in playoff_8:
            top4_count[t] += 1

    results = {}
    for team in all_teams:
        results[team] = {
            "champion_prob":  round(champion_count[team] / n * 100, 1),
            "top4_prob":      round(top4_count[team] / n * 100, 1),
            "is_direct_seed": team in PLAYOFF_DIRECT,
        }

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  VCT Masters London 2026 — Monte Carlo Simulation")
    print("=" * 60)

    # Load model
    try:
        saved        = joblib.load(f"{MODEL_DIR}/vct2025_model.pkl")
        model        = saved["model"]
        feature_cols = saved["features"]
        team_stats   = pd.read_csv(f"{DATA_DIR}/vct2025_team_stats.csv")
        print("✓ Model loaded")
    except Exception as e:
        print(f"[!] Model tidak ditemukan ({e}), pakai Elo-based estimate")
        model = None; feature_cols = None; team_stats = pd.DataFrame()

    # Build win probability matrix
    all_teams = SWISS_TEAMS + PLAYOFF_DIRECT
    win_probs = {}
    for t1, t2 in combinations(all_teams, 2):
        p = get_win_probability(t1, t2, model, feature_cols, team_stats)
        win_probs[(t1, t2)] = p

    print(f"\n✓ Win probability matrix: {len(win_probs)} matchups")
    print("\nSample win probs (team1 vs team2 → team1 win%):")
    samples = [("G2 Esports","NRG"), ("Paper Rex","Team Vitality"),
               ("NRG","Team Heretics"), ("EDward Gaming","FULL SENSE")]
    for t1, t2 in samples:
        p = win_probs.get((t1,t2), 1 - win_probs.get((t2,t1), 0.5))
        print(f"  {t1:<25} vs {t2:<25} → {p*100:.1f}%")

    # Swiss stage advance probability
    print("\n[1/2] Simulasi Swiss stage (10,000x)...")
    advance_counts, n = run_swiss_simulation(win_probs, n_simulations=10000)
    print("\nProbabilitas lolos Swiss stage:")
    for team in sorted(SWISS_TEAMS, key=lambda t: -advance_counts[t]):
        prob = advance_counts[team] / n * 100
        bar  = "█" * int(prob / 5)
        print(f"  {team:<30} {bar:<20} {prob:.1f}%")

    # Full simulation
    print("\n[2/2] Simulasi penuh Swiss + Playoff (10,000x)...")
    results = run_full_simulation(win_probs, n=10000)

    print("\n=== PROBABILITAS JUARA MASTERS LONDON 2026 ===")
    sorted_results = sorted(results.items(), key=lambda x: -x[1]["champion_prob"])
    for team, data in sorted_results:
        seed_badge = "[DIRECT]" if data["is_direct_seed"] else "[SWISS] "
        bar = "█" * int(data["champion_prob"] / 2)
        print(f"  {seed_badge} {team:<28} {bar:<25} {data['champion_prob']:.1f}%")

    # Simpan hasil
    out = {
        "swiss_advance_prob": {t: round(advance_counts[t]/n*100, 1) for t in SWISS_TEAMS},
        "champion_prob":      {t: results[t]["champion_prob"] for t in results},
        "top4_prob":          {t: results[t]["top4_prob"] for t in results},
        "win_prob_matrix":    {f"{t1}|{t2}": round(p*100,1) for (t1,t2),p in win_probs.items()},
        "n_simulations":      10000,
    }
    out_path = f"{DATA_DIR}/simulation_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Hasil disimpan ke {out_path}")
    print("\nJalankan dashboard untuk lihat visualisasi lengkap:")
    print("  streamlit run dashboard.py")

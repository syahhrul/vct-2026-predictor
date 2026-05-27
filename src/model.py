"""
Model Training & Prediksi — VCT 2025 (v3)
"""

import pandas as pd
import numpy as np
import json, os, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

BASE_FEATURES = [
    "win_rate_diff", "form_diff", "elo_diff",
    "h2h_t1_winrate", "t1_win_rate", "t2_win_rate",
    "t1_recent_form", "t2_recent_form", "t1_elo", "t2_elo",
]
OPTIONAL_FEATURES = ["acs_diff", "rating_diff", "t1_avg_acs", "t2_avg_acs"]


def load_features():
    df = pd.read_csv(f"{DATA_DIR}/vct2025_features.csv")
    df = df.dropna(subset=["label"])

    feature_cols = BASE_FEATURES.copy()
    for col in OPTIONAL_FEATURES:
        if col in df.columns and df[col].notna().sum() > 10:
            feature_cols.append(col)

    print(f"  Features    : {feature_cols}")
    null_counts = df[feature_cols].isnull().sum()
    if null_counts.any():
        print(f"  NaN counts  :\n{null_counts[null_counts > 0].to_string()}")
    return df, feature_cols


def build_pipeline():
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                    learning_rate=0.05, random_state=42)
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("gb", gb)],
        voting="soft", weights=[1, 2],
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   ensemble),
    ])


def evaluate(pipeline, X, y):
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    auc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")

    print(f"\n  CV Accuracy : {acc.mean():.4f} ± {acc.std():.4f}")
    print(f"  CV ROC-AUC  : {auc.mean():.4f} ± {auc.std():.4f}")

    pipeline.fit(X, y)
    probs = pipeline.predict_proba(X)[:, 1]
    brier = brier_score_loss(y, probs)
    print(f"  Brier Score : {brier:.4f}  (0.25 = random, lower is better)")

    return {
        "cv_accuracy_mean": round(acc.mean(), 4),
        "cv_accuracy_std":  round(acc.std(), 4),
        "cv_roc_auc_mean":  round(auc.mean(), 4),
        "cv_roc_auc_std":   round(auc.std(), 4),
        "brier_score":      round(brier, 4),
        "n_samples":        len(y),
        "features_used":    list(X.columns),
    }


def train_and_save(features_df, feature_cols):
    print("\n[2/3] Training model...")
    X = features_df[feature_cols]
    y = features_df["label"]

    pipeline = build_pipeline()
    metrics  = evaluate(pipeline, X, y)

    with open(f"{MODEL_DIR}/vct2025_model.pkl", "wb") as f:
        pickle.dump({"model": pipeline, "features": feature_cols}, f)
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  -> Model disimpan ke models/vct2025_model.pkl")
    return pipeline, feature_cols


def predict_match(team1, team2, pipeline=None, feature_cols=None):
    """
    Prediksi satu match. Load model otomatis kalau tidak di-pass.
    Menggunakan Elo final dari team_stats.
    """
    if pipeline is None:
        with open(f"{MODEL_DIR}/vct2025_model.pkl", "rb") as f:
            saved = pickle.load(f)
        pipeline     = saved["model"]
        feature_cols = saved["features"]

    team_stats = pd.read_csv(f"{DATA_DIR}/vct2025_team_stats.csv").set_index("team")

    player_perf = None
    pp_path = f"{DATA_DIR}/vct2025_team_player_perf.csv"
    if os.path.exists(pp_path):
        pp_df = pd.read_csv(pp_path)
        if not pp_df.empty and pp_df["avg_acs"].notna().any():
            player_perf = pp_df.set_index("team")

    for team in [team1, team2]:
        if team not in team_stats.index:
            return {"error": f"Tim tidak ditemukan: {team}"}

    feat = {
        "win_rate_diff":  team_stats.loc[team1, "win_rate"]      - team_stats.loc[team2, "win_rate"],
        "form_diff":      team_stats.loc[team1, "recent_form_5"] - team_stats.loc[team2, "recent_form_5"],
        "elo_diff":       team_stats.loc[team1, "elo_final"]     - team_stats.loc[team2, "elo_final"],
        "h2h_t1_winrate": 0.5,
        "t1_win_rate":    team_stats.loc[team1, "win_rate"],
        "t2_win_rate":    team_stats.loc[team2, "win_rate"],
        "t1_recent_form": team_stats.loc[team1, "recent_form_5"],
        "t2_recent_form": team_stats.loc[team2, "recent_form_5"],
        "t1_elo":         team_stats.loc[team1, "elo_final"],
        "t2_elo":         team_stats.loc[team2, "elo_final"],
    }

    if player_perf is not None and team1 in player_perf.index and team2 in player_perf.index:
        feat["acs_diff"]    = player_perf.loc[team1, "avg_acs"]    - player_perf.loc[team2, "avg_acs"]
        feat["rating_diff"] = player_perf.loc[team1, "avg_rating"] - player_perf.loc[team2, "avg_rating"]
        feat["t1_avg_acs"]  = player_perf.loc[team1, "avg_acs"]
        feat["t2_avg_acs"]  = player_perf.loc[team2, "avg_acs"]

    X = pd.DataFrame([feat]).reindex(columns=feature_cols)
    probs = pipeline.predict_proba(X)[0]

    return {
        "team1":            team1,
        "team2":            team2,
        "prob_team1_win":   round(float(probs[1]) * 100, 1),
        "prob_team2_win":   round(float(probs[0]) * 100, 1),
        "predicted_winner": team1 if probs[1] > 0.5 else team2,
        "confidence":       round(max(probs) * 100, 1),
    }


if __name__ == "__main__":
    print("=" * 55)
    print("  VCT 2025 Model Training (v3 - temporal)")
    print("=" * 55)

    print("\n[1/3] Loading features...")
    features_df, feature_cols = load_features()
    print(f"  Dataset     : {len(features_df)} matches")
    print(f"  Label dist  :\n{features_df['label'].value_counts().to_string()}")

    pipeline, feature_cols = train_and_save(features_df, feature_cols)

    print("\n[3/3] Contoh prediksi matchup terkenal...")
    matchups = [
        ("NRG", "FNATIC"),
        ("G2 Esports", "Paper Rex"),
        ("Sentinels", "MIBR"),
        ("Team Liquid", "FNATIC"),
    ]
    for t1, t2 in matchups:
        res = predict_match(t1, t2, pipeline, feature_cols)
        if "error" not in res:
            bar1 = "█" * int(res["prob_team1_win"] / 5)
            bar2 = "█" * int(res["prob_team2_win"] / 5)
            print(f"\n  {t1:20} vs {t2}")
            print(f"  {bar1:<20} {res['prob_team1_win']}%")
            print(f"  {bar2:<20} {res['prob_team2_win']}%")
            print(f"  → Predicted: {res['predicted_winner']} ({res['confidence']}% confidence)")
        else:
            print(f"\n  [{t1} vs {t2}] {res['error']}")

    print("\n✅ Selesai!")

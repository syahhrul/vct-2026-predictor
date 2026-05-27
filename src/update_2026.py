"""
VCT 2026 Scraper — fresh start, data 2025 tidak dipakai
Jalankan: python src/update_2026.py
Lalu    : python src/feature_engineering.py && python src/model.py
"""

import sys, os, json, re, time
sys.path.insert(0, os.path.dirname(__file__))

import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://www.vlr.gg"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

VCT_2026_EVENTS = [
    {"name": "VCT 2026: Americas Kickoff",     "id": "2682", "region": "Americas"},
    {"name": "VCT 2026: EMEA Kickoff",         "id": "2684", "region": "EMEA"},
    {"name": "VCT 2026: Pacific Kickoff",      "id": "2683", "region": "Pacific"},
    {"name": "VCT 2026: China Kickoff",        "id": "2685", "region": "China"},
    {"name": "Valorant Masters Santiago 2026", "id": "2760", "region": "International"},
    {"name": "VCT 2026: Americas Stage 1",     "id": "2860", "region": "Americas"},
    {"name": "VCT 2026: EMEA Stage 1",         "id": "2863", "region": "EMEA"},
    {"name": "VCT 2026: Pacific Stage 1",      "id": "2775", "region": "Pacific"},
    {"name": "VCT 2026: China Stage 1",        "id": "2864", "region": "China"},
]


def clean(s):
    return re.sub(r'\s+', ' ', s).strip()


def get_soup(url, delay=1.5):
    try:
        time.sleep(delay)
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml")
    except Exception as e:
        print(f"  [ERROR] {url} -> {e}")
        return None


# ─────────────────────────────────────────────
# SCRAPE MATCHES
# ─────────────────────────────────────────────

def scrape_event_matches(event_id, event_name):
    slug = event_name.lower().replace(" ", "-").replace(":", "").replace("'", "")
    url  = f"{BASE_URL}/event/matches/{event_id}/{slug}/?series_id=all"
    soup = get_soup(url)
    if not soup:
        return []

    matches = []
    for a in soup.find_all("a", href=True):
        href  = a["href"]
        parts = href.strip("/").split("/")
        if len(parts) < 2 or not parts[0].isdigit() or "vs" not in parts[1]:
            continue

        teams  = a.select(".match-item-vs-team-name")
        scores = a.select(".match-item-vs-team-score")
        date   = a.select_one(".match-item-time")
        stage  = a.select_one(".match-item-event-series")
        status = a.select_one(".ml-status")

        if len(teams) < 2:
            continue

        matches.append({
            "event":    event_name,
            "event_id": event_id,
            "match_id": parts[0],
            "team1":    clean(teams[0].text),
            "team2":    clean(teams[1].text),
            "score1":   clean(scores[0].text) if len(scores) > 0 else "",
            "score2":   clean(scores[1].text) if len(scores) > 1 else "",
            "date":     clean(date.text)   if date   else "",
            "stage":    clean(stage.text)  if stage  else "",
            "status":   clean(status.text) if status else "",
            "url":      BASE_URL + href,
        })
    return matches


# ─────────────────────────────────────────────
# SCRAPE MATCH DETAIL
# ─────────────────────────────────────────────

def scrape_match_detail(match_url, match_id):
    soup = get_soup(match_url, delay=2.0)
    if not soup:
        return {}

    result = {"match_id": match_id, "url": match_url, "maps": []}

    for block in soup.select(".vm-stats-game"):
        if block.get("data-game-id") == "all":
            continue

        map_div  = block.select_one(".map")
        if map_div:
            for b in map_div.select(".mod-played,.mod-pick"):
                b.decompose()
            map_name = re.sub(r'\s*(PICK|BAN|REMAIN)\s*$', '',
                              clean(map_div.get_text()), flags=re.IGNORECASE).strip()
        else:
            map_name = "Unknown"

        sc  = block.select(".score")
        ts1 = clean(sc[0].text) if len(sc) > 0 else ""
        ts2 = clean(sc[1].text) if len(sc) > 1 else ""

        players = []
        for row in block.select("tbody tr"):
            cols = row.select("td")
            if len(cols) < 8:
                continue
            name_el = cols[0].select_one(".text-of")
            imgs    = cols[1].select("img")
            agent   = imgs[0].get("alt", "").strip() if imgs else ""

            def first_num(t):
                nums = re.findall(r'[\d.]+', clean(t))
                return nums[0] if nums else ""

            kda   = clean(cols[4].text) if len(cols) > 4 else ""
            nums  = re.findall(r'\d+', kda)
            kills, deaths, assists = (nums[0], nums[1], nums[2]) if len(nums) >= 3 else ("","","")

            players.append({
                "player":  clean(name_el.text) if name_el else "",
                "agent":   agent,
                "rating":  first_num(cols[2].text) if len(cols) > 2 else "",
                "acs":     first_num(cols[3].text) if len(cols) > 3 else "",
                "kills":   kills,
                "deaths":  deaths,
                "assists": assists,
                "adr":     first_num(cols[7].text) if len(cols) > 7 else "",
                "hs_pct":  clean(cols[8].text).replace("%","") if len(cols) > 8 else "",
                "fk":      first_num(cols[9].text) if len(cols) > 9 else "",
                "fd":      first_num(cols[10].text) if len(cols) > 10 else "",
            })

        if players:
            result["maps"].append({
                "map": map_name, "team1_score": ts1,
                "team2_score": ts2, "players": players,
            })
    return result


# ─────────────────────────────────────────────
# FLATTEN PLAYER STATS
# ─────────────────────────────────────────────

def flatten_player_stats(details):
    rows = []
    for m in details:
        for md in m.get("maps", []):
            for p in md.get("players", []):
                rows.append({
                    "match_id": m.get("match_id"), "event": m.get("event"),
                    "team1": m.get("team1"), "team2": m.get("team2"),
                    "match_score1": m.get("score1"), "match_score2": m.get("score2"),
                    "map": md.get("map"),
                    "map_score_t1": md.get("team1_score"),
                    "map_score_t2": md.get("team2_score"),
                    **p,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ["rating","acs","kills","deaths","assists","adr","hs_pct","fk","fd"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  VCT 2026 Data Scraper — fresh start")
    print("=" * 55)

    # ── Step 1: Events
    print(f"\n[1/4] Events ({len(VCT_2026_EVENTS)} total):")
    for ev in VCT_2026_EVENTS:
        print(f"     . {ev['name']} ({ev['region']})")
    pd.DataFrame(VCT_2026_EVENTS).to_csv(f"{DATA_DIR}/vct2025_events.csv", index=False)

    # ── Step 2: Matches
    print("\n[2/4] Scraping matches...")
    all_matches = []
    for ev in tqdm(VCT_2026_EVENTS, desc="Events"):
        m = scrape_event_matches(ev["id"], ev["name"])
        all_matches.extend(m)
        tqdm.write(f"  {ev['name']}: {len(m)} matches")

    df = pd.DataFrame(all_matches).drop_duplicates(subset=["match_id"])
    df.to_csv(f"{DATA_DIR}/vct2025_matches.csv", index=False)
    print(f"\n  -> {len(df)} total matches")

    # ── Step 3: Match details
    completed = df[
        df["score1"].notna() &
        (df["score1"].astype(str).str.strip() != "") &
        (df["score1"].astype(str).str.strip() != "--")
    ]
    print(f"\n[3/4] Scraping {len(completed)} match details...")

    details = []
    for _, row in tqdm(completed.iterrows(), total=len(completed), desc="Details"):
        d = scrape_match_detail(row["url"], str(row["match_id"]))
        if d and d.get("maps"):
            d.update({
                "team1": row["team1"], "team2": row["team2"],
                "score1": row["score1"], "score2": row["score2"],
                "event": row["event"],
            })
            details.append(d)

    with open(f"{DATA_DIR}/vct2025_match_details.json", "w") as f:
        json.dump(details, f, indent=2)
    print(f"  -> {len(details)} match details")

    # ── Step 4: Player stats
    print("\n[4/4] Flattening player stats...")
    ps = flatten_player_stats(details)
    if not ps.empty:
        ps.to_csv(f"{DATA_DIR}/vct2025_player_stats.csv", index=False)
        print(f"  -> {len(ps)} rows")
        print(f"\n  Null check:")
        print(ps[["rating","acs","kills","deaths","adr"]].isnull().sum().to_string())

    print("\nFile output:")
    for f in ["vct2025_events.csv","vct2025_matches.csv",
              "vct2025_match_details.json","vct2025_player_stats.csv"]:
        path = f"{DATA_DIR}/{f}"
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"  {'OK' if size > 0 else 'MISSING'} data/{f} ({size:,} bytes)")

    print("\n✅ Selesai! Jalankan selanjutnya:")
    print("  python src/feature_engineering.py")
    print("  python src/model.py")

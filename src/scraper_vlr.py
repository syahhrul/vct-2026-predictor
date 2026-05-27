"""
VLR.gg Scraper — VCT 2025 (v4 - fixed map name & stats parsing)
"""

import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
import os
from tqdm import tqdm

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

BASE_URL = "https://www.vlr.gg"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

VCT_2025_EVENTS = [
    {"name": "Valorant Champions 2025",       "id": "2283", "region": "International"},
    {"name": "VCT 2025: Americas Stage 2",    "id": "2501", "region": "Americas"},
    {"name": "VCT 2025: EMEA Stage 2",        "id": "2498", "region": "EMEA"},
    {"name": "VCT 2025: Pacific Stage 2",     "id": "2500", "region": "Pacific"},
    {"name": "VCT 2025: China Stage 2",       "id": "2499", "region": "China"},
    {"name": "Valorant Masters Toronto 2025", "id": "2420", "region": "International"},
    {"name": "VCT 2025: EMEA Stage 1",        "id": "2380", "region": "EMEA"},
    {"name": "VCT 2025: Pacific Stage 1",     "id": "2347", "region": "Pacific"},
    {"name": "VCT 2025: Americas Stage 1",    "id": "2346", "region": "Americas"},
    {"name": "Valorant Masters Bangkok 2025", "id": "2274", "region": "International"},
]


def get_soup(url: str, delay: float = 1.5):
    try:
        time.sleep(delay)
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"  [ERROR] {url} -> {e}")
        return None


def clean_text(s: str) -> str:
    """Hapus whitespace berlebih, tab, newline."""
    return re.sub(r'\s+', ' ', s).strip()


# ─────────────────────────────────────────────
# 1. EVENTS
# ─────────────────────────────────────────────

def save_events():
    print("\n[1/4] VCT 2025 events...")
    df = pd.DataFrame(VCT_2025_EVENTS)
    df.to_csv(f"{DATA_DIR}/vct2025_events.csv", index=False)
    for ev in VCT_2025_EVENTS:
        print(f"     . {ev['name']} ({ev['region']})")
    print(f"  -> {len(VCT_2025_EVENTS)} events disimpan")
    return VCT_2025_EVENTS


# ─────────────────────────────────────────────
# 2. MATCHES
# ─────────────────────────────────────────────

def scrape_event_matches(event_id: str, event_name: str):
    slug = event_name.lower().replace(" ", "-").replace(":", "").replace("'", "")
    url = f"{BASE_URL}/event/matches/{event_id}/{slug}/?series_id=all"

    soup = get_soup(url)
    if not soup:
        return []

    matches = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        parts = href.strip("/").split("/")
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        if "vs" not in parts[1]:
            continue

        match_id = parts[0]
        team_els  = a_tag.select(".match-item-vs-team-name")
        score_els = a_tag.select(".match-item-vs-team-score")
        date_el   = a_tag.select_one(".match-item-time")
        stage_el  = a_tag.select_one(".match-item-event-series")
        status_el = a_tag.select_one(".ml-status")

        if len(team_els) < 2:
            continue

        matches.append({
            "event":    event_name,
            "event_id": event_id,
            "match_id": match_id,
            "team1":    clean_text(team_els[0].text),
            "team2":    clean_text(team_els[1].text),
            "score1":   clean_text(score_els[0].text) if len(score_els) > 0 else "",
            "score2":   clean_text(score_els[1].text) if len(score_els) > 1 else "",
            "date":     clean_text(date_el.text)   if date_el   else "",
            "stage":    clean_text(stage_el.text)  if stage_el  else "",
            "status":   clean_text(status_el.text) if status_el else "",
            "url":      BASE_URL + href,
        })

    return matches


def scrape_all_matches(events):
    print("\n[2/4] Scraping match results...")
    all_matches = []

    for ev in tqdm(events, desc="Events"):
        matches = scrape_event_matches(ev["id"], ev["name"])
        all_matches.extend(matches)
        tqdm.write(f"  {ev['name']}: {len(matches)} matches")

    df = pd.DataFrame(all_matches)
    if df.empty:
        print("  [!] Tidak ada match.")
        return df

    df = df.drop_duplicates(subset=["match_id"])
    print(f"\n  Sample:\n{df[['event','team1','team2','score1','score2','status']].head(5).to_string()}")
    df.to_csv(f"{DATA_DIR}/vct2025_matches.csv", index=False)
    print(f"\n  -> Total {len(df)} matches -> data/vct2025_matches.csv")
    return df


# ─────────────────────────────────────────────
# 3. MATCH DETAIL (per-map player stats)
# ─────────────────────────────────────────────

def parse_kda_cell(text: str):
    """
    Cell K/D/A di vlr.gg formatnya: '20 / 15 / 3' (all maps)
    atau hanya angka untuk satu map.
    Ambil nilai pertama (all maps aggregate).
    """
    text = clean_text(text)
    # Format: "20 / 15 / 3" → ambil split by '/'
    parts = [p.strip() for p in text.split("/")]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]  # kills, deaths, assists
    # Kadang hanya satu angka
    return text, "", ""


def scrape_match_detail(match_url: str, match_id: str):
    soup = get_soup(match_url, delay=2.0)
    if not soup:
        return {}

    result = {"match_id": match_id, "url": match_url, "maps": []}

    for game_block in soup.select(".vm-stats-game"):
        if game_block.get("data-game-id") == "all":
            continue

        # ── Map name ──────────────────────────────────────────
        # Ambil teks dari .map div, bersihkan whitespace/tab/newline
        map_div = game_block.select_one(".map")
        if map_div:
            # Hapus child span seperti "PICK" badge
            for badge in map_div.select(".mod-played, .mod-pick"):
                badge.decompose()
            map_name = clean_text(map_div.get_text())
        else:
            map_name = "Unknown"

        # Hapus sisa "PICK" / "BAN" di akhir
        map_name = re.sub(r'\s*(PICK|BAN|REMAIN)\s*$', '', map_name, flags=re.IGNORECASE).strip()

        # ── Scores ────────────────────────────────────────────
        score_els = game_block.select(".score")
        team_score1 = clean_text(score_els[0].text) if len(score_els) > 0 else ""
        team_score2 = clean_text(score_els[1].text) if len(score_els) > 1 else ""

        # ── Players ───────────────────────────────────────────
        # Struktur tabel: R | ACS | K/D/A | +/- | KAST | ADR | HS% | FK | FD | FK-FD
        # K, D, A ada dalam SATU cell dengan format "K / D / A"
        players = []
        for row in game_block.select("tbody tr"):
            cols = row.select("td")
            if len(cols) < 9:
                continue

            name_el    = cols[0].select_one(".text-of")
            agent_imgs = cols[1].select("img")
            agent      = agent_imgs[0].get("alt", "").strip() if agent_imgs else ""

            # col index: 0=player, 1=agent, 2=R, 3=ACS, 4=K/D/A, 5=+/-, 6=KAST, 7=ADR, 8=HS%, 9=FK, 10=FD
            rating_text = clean_text(cols[2].text) if len(cols) > 2 else ""
            acs_text    = clean_text(cols[3].text) if len(cols) > 3 else ""
            kda_text    = clean_text(cols[4].text) if len(cols) > 4 else ""
            adr_text    = clean_text(cols[7].text) if len(cols) > 7 else ""
            hs_text     = clean_text(cols[8].text) if len(cols) > 8 else ""
            fk_text     = clean_text(cols[9].text) if len(cols) > 9 else ""
            fd_text     = clean_text(cols[10].text) if len(cols) > 10 else ""

            # Parse K/D/A — ambil nilai "all maps" (pertama) jika ada spasi
            # Format bisa: "20 8 12" (spasi) atau "20 / 15 / 3" (dengan slash)
            # Di HTML sebenarnya ada sub-span per map, ambil teks penuh lalu split
            kills = deaths = assists = ""
            kda_clean = re.sub(r'\s+', ' ', kda_text).strip()

            # Coba parse format "K D A" (spasi) — ambil 3 angka pertama
            numbers = re.findall(r'\d+', kda_clean)
            if len(numbers) >= 3:
                kills, deaths, assists = numbers[0], numbers[1], numbers[2]

            # Ambil angka pertama saja untuk rating, acs, adr (bisa ada "all/attack/defend")
            def first_num(t):
                nums = re.findall(r'[\d.]+', t)
                return nums[0] if nums else ""

            players.append({
                "player":  clean_text(name_el.text) if name_el else "",
                "agent":   agent,
                "rating":  first_num(rating_text),
                "acs":     first_num(acs_text),
                "kills":   kills,
                "deaths":  deaths,
                "assists": assists,
                "adr":     first_num(adr_text),
                "hs_pct":  hs_text.replace("%", "").strip(),
                "fk":      first_num(fk_text),
                "fd":      first_num(fd_text),
            })

        if players:
            result["maps"].append({
                "map":         map_name,
                "team1_score": team_score1,
                "team2_score": team_score2,
                "players":     players,
            })

    return result


def scrape_match_details_batch(matches_df: pd.DataFrame, max_matches: int = 100):
    print(f"\n[3/4] Scraping detail {max_matches} matches...")

    completed_mask = (
        matches_df["score1"].notna() &
        matches_df["score2"].notna() &
        (matches_df["score1"].astype(str).str.strip() != "") &
        (matches_df["score1"].astype(str).str.strip() != "--")
    )
    completed = matches_df[completed_mask].head(max_matches)
    print(f"  -> {len(completed)} completed matches")

    if completed.empty:
        print("  [!] Tidak ada completed matches.")
        return []

    details = []
    for _, row in tqdm(completed.iterrows(), total=len(completed), desc="Details"):
        detail = scrape_match_detail(row["url"], str(row["match_id"]))
        if detail and detail.get("maps"):
            detail.update({
                "team1":  row["team1"],
                "team2":  row["team2"],
                "score1": row["score1"],
                "score2": row["score2"],
                "event":  row["event"],
            })
            details.append(detail)

    with open(f"{DATA_DIR}/vct2025_match_details.json", "w") as f:
        json.dump(details, f, indent=2)

    print(f"  -> {len(details)} match details -> data/vct2025_match_details.json")
    return details


# ─────────────────────────────────────────────
# 4. FLATTEN
# ─────────────────────────────────────────────

def flatten_player_stats(match_details):
    print("\n[4/4] Flattening player stats...")
    rows = []

    for match in match_details:
        for map_data in match.get("maps", []):
            for player in map_data.get("players", []):
                rows.append({
                    "match_id":     match.get("match_id"),
                    "event":        match.get("event"),
                    "team1":        match.get("team1"),
                    "team2":        match.get("team2"),
                    "match_score1": match.get("score1"),
                    "match_score2": match.get("score2"),
                    "map":          map_data.get("map"),
                    "map_score_t1": map_data.get("team1_score"),
                    "map_score_t2": map_data.get("team2_score"),
                    **player,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  [!] Tidak ada data.")
        return df

    for col in ["rating", "acs", "kills", "deaths", "assists", "adr", "hs_pct", "fk", "fd"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_csv(f"{DATA_DIR}/vct2025_player_stats.csv", index=False)
    print(f"  -> {len(df)} rows -> data/vct2025_player_stats.csv")
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  VCT 2025 Data Scraper — vlr.gg (v4)")
    print("=" * 55)

    events = save_events()
    matches_df = scrape_all_matches(events)
    if matches_df.empty:
        exit(1)

    # Test dengan 30 match dulu
    details = scrape_match_details_batch(matches_df, max_matches=30)

    if details:
        player_df = flatten_player_stats(details)
        if not player_df.empty:
            print(f"\nPreview player stats:")
            print(player_df[["event","team1","team2","map","player","rating","acs","kills","deaths","adr"]].head(10).to_string())
            print(f"\nNull count per kolom:")
            print(player_df[["rating","acs","kills","deaths","adr"]].isnull().sum())

    print("\nFile output:")
    for f in ["vct2025_events.csv","vct2025_matches.csv",
              "vct2025_match_details.json","vct2025_player_stats.csv"]:
        path = f"{DATA_DIR}/{f}"
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"  {'OK' if size > 0 else 'MISSING'} data/{f} ({size:,} bytes)")

# ─────────────────────────────────────────────
# ENTRY POINT: scrape semua (untuk produksi)
# ─────────────────────────────────────────────
def scrape_full():
    """Scrape semua match + semua detail. Jalankan ini untuk data lengkap."""
    events     = save_events()
    matches_df = scrape_all_matches(events)
    if matches_df.empty:
        return
    details    = scrape_match_details_batch(matches_df, max_matches=len(matches_df))
    if details:
        flatten_player_stats(details)

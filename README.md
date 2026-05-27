# VCT 2026 Match Predictor

Prediksi hasil match Valorant Champions Tour 2026 berdasarkan data dari vlr.gg.

## Struktur Project

```
vct_predictor/
├── data/                    ← CSV & JSON hasil scraping (auto-generated)
├── models/                  ← Model yang tersimpan (auto-generated)
├── src/
│   ├── scraper_vlr.py       ← Step 1: Ambil data dari vlr.gg
│   ├── feature_engineering.py ← Step 2: Hitung fitur untuk ML
│   └── model.py             ← Step 3: Train & prediksi
└── requirements.txt
```

## Setup

```bash
# 1. Buat virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

## Cara Menjalankan (urut)

### Step 1 — Scraping Data
```bash
python src/scraper_vlr.py
```
Estimasi waktu: 15–30 menit (tergantung jumlah match)

Output:
- `data/vct2026_events.csv`
- `data/vct2026_matches.csv`
- `data/vct2026_match_details.json`
- `data/vct2026_player_stats.csv`

### Step 2 — Feature Engineering
```bash
python src/feature_engineering.py
```
Output:
- `data/vct2026_features.csv`
- `data/vct2026_team_stats.csv`
- `data/vct2026_team_player_perf.csv`

### Step 3 — Training Model
```bash
python src/model.py
```
Output:
- `models/vct2026_model.pkl`
- `models/metrics.json`

## Fitur yang Digunakan Model

| Fitur | Deskripsi |
|-------|-----------|
| `elo_diff` | Selisih Elo rating (proxy kekuatan tim) |
| `win_rate_diff` | Selisih win rate keseluruhan |
| `form_diff` | Selisih recent form (5 match terakhir) |
| `h2h_t1_winrate` | Win rate head-to-head |
| `acs_diff` | Selisih rata-rata ACS (firepower) |
| `rating_diff` | Selisih rata-rata rating VLR |

## Tips

- Jalankan ulang scraper secara berkala untuk data terbaru
- Untuk `max_matches` di scraper, mulai dari 50 dulu, setelah yakin baru naik ke 200+
- Kalau mau scraping lebih cepat, kurangi `delay` di `get_soup()` — tapi risiko di-rate-limit vlr.gg

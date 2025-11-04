import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import re

st.set_page_config(page_title="NFL Game + Player Props Dashboard", layout="wide")

# =========================
# Data Sources (Google Sheets)
# =========================
SCORE_URL = "https://docs.google.com/spreadsheets/d/1KrTQbR5uqlBn2v2Onpjo6qHFnLlrqIQBzE52KAhMYcY/export?format=csv"

SHEETS = {
    "total_offense": "https://docs.google.com/spreadsheets/d/1DFZRqOiMXbIoEeLaNaWh-4srxeWaXscqJxIAHt9yq48/export?format=csv",
    "total_passing": "https://docs.google.com/spreadsheets/d/1QclB5ajymBsCC09j8s4Gie_bxj4ebJwEw4kihG6uCng/export?format=csv",
    "total_rushing": "https://docs.google.com/spreadsheets/d/14NgUntobNrL1AZg3U85yZInArFkHyf9mi1csVFodu90/export?format=csv",
    "total_scoring": "https://docs.google.com/spreadsheets/d/1SJ_Y1ljU44lOjbNHuXGyKGiF3mgQxjAjX8H3j-CCqSw/export?format=csv",
    "player_receiving": "https://docs.google.com/spreadsheets/d/1Gwb2A-a4ge7UKHnC7wUpJltgioTuCQNuwOiC5ecZReM/export?format=csv",
    "player_rushing": "https://docs.google.com/spreadsheets/d/1c0xpi_wZSf8VhkSPzzchxvhzAQHK0tFetakdRqb3e6k/export?format=csv",
    "player_passing": "https://docs.google.com/spreadsheets/d/1I9YNSQMylW_waJs910q4S6SM8CZE--hsyNElrJeRfvk/export?format=csv",
    "def_rb": "https://docs.google.com/spreadsheets/d/1xTP8tMnEVybu9vYuN4i6IIrI71q1j60BuqVC40fjNeY/export?format=csv",
    "def_qb": "https://docs.google.com/spreadsheets/d/1SEwUdExz7Px61FpRNQX3bUsxVFtK97JzuQhTddVa660/export?format=csv",
    "def_wr": "https://docs.google.com/spreadsheets/d/14klXrrHHCLlXhW6-F-9eJIz3dkp_ROXVSeehlM8TYAo/export?format=csv",
    "def_te": "https://docs.google.com/spreadsheets/d/1yMpgtx1ObYLDVufTMR5Se3KrMi1rG6UzMzLcoptwhi4/export?format=csv",
}

# =========================
# Team alias map (handles full names, nicknames, abbreviations)
# =========================
TEAM_ALIAS_TO_CODE = {
    # Cardinals
    "arizona cardinals": "ARI", "cardinals": "ARI", "arizona": "ARI", "ari": "ARI",
    # Falcons
    "atlanta falcons": "ATL", "falcons": "ATL", "atlanta": "ATL", "atl": "ATL",
    # Ravens
    "baltimore ravens": "BAL", "ravens": "BAL", "baltimore": "BAL", "bal": "BAL",
    # Bills
    "buffalo bills": "BUF", "bills": "BUF", "buffalo": "BUF", "buf": "BUF",
    # Panthers
    "carolina panthers": "CAR", "panthers": "CAR", "carolina": "CAR", "car": "CAR",
    # Bears
    "chicago bears": "CHI", "bears": "CHI", "chicago": "CHI", "chi": "CHI",
    # Bengals
    "cincinnati bengals": "CIN", "bengals": "CIN", "cincinnati": "CIN", "cin": "CIN",
    # Browns
    "cleveland browns": "CLE", "browns": "CLE", "cleveland": "CLE", "cle": "CLE",
    # Cowboys
    "dallas cowboys": "DAL", "cowboys": "DAL", "dallas": "DAL", "dal": "DAL",
    # Broncos
    "denver broncos": "DEN", "broncos": "DEN", "denver": "DEN", "den": "DEN",
    # Lions
    "detroit lions": "DET", "lions": "DET", "detroit": "DET", "det": "DET",
    # Packers
    "green bay packers": "GB", "packers": "GB", "green bay": "GB", "gb": "GB",
    # Texans
    "houston texans": "HOU", "texans": "HOU", "houston": "HOU", "hou": "HOU",
    # Colts
    "indianapolis colts": "IND", "colts": "IND", "indianapolis": "IND", "ind": "IND",
    # Jaguars
    "jacksonville jaguars": "JAX", "jaguars": "JAX", "jacksonville": "JAX", "jax": "JAX", "jacs": "JAX",
    # Chiefs
    "kansas city chiefs": "KC", "chiefs": "KC", "kansas city": "KC", "kc": "KC",
    # Raiders (incl. legacy)
    "las vegas raiders": "LV", "raiders": "LV", "las vegas": "LV", "lv": "LV",
    "oakland raiders": "LV", "oakland": "LV",
    # Chargers (incl. legacy)
    "los angeles chargers": "LAC", "la chargers": "LAC", "chargers": "LAC", "lac": "LAC",
    "san diego chargers": "LAC", "san diego": "LAC",
    # Rams (incl. legacy)
    "los angeles rams": "LAR", "la rams": "LAR", "rams": "LAR", "lar": "LAR",
    "st. louis rams": "LAR", "st louis rams": "LAR", "st louis": "LAR",
    # Dolphins
    "miami dolphins": "MIA", "dolphins": "MIA", "miami": "MIA", "mia": "MIA",
    # Vikings
    "minnesota vikings": "MIN", "vikings": "MIN", "minnesota": "MIN", "min": "MIN",
    # Patriots
    "new england patriots": "NE", "patriots": "NE", "new england": "NE", "ne": "NE",
    # Saints
    "new orleans saints": "NO", "saints": "NO", "new orleans": "NO", "no": "NO", "nos": "NO",
    # Giants
    "new york giants": "NYG", "ny giants": "NYG", "giants": "NYG", "nyg": "NYG",
    # Jets
    "new york jets": "NYJ", "ny jets": "NYJ", "jets": "NYJ", "nyj": "NYJ",
    # Eagles
    "philadelphia eagles": "PHI", "eagles": "PHI", "philadelphia": "PHI", "phi": "PHI",
    # Steelers
    "pittsburgh steelers": "PIT", "steelers": "PIT", "pittsburgh": "PIT", "pit": "PIT",
    # 49ers
    "san francisco 49ers": "SF", "49ers": "SF", "niners": "SF", "san francisco": "SF", "sf": "SF",
    # Seahawks
    "seattle seahawks": "SEA", "seahawks": "SEA", "seattle": "SEA", "sea": "SEA",
    # Buccaneers
    "tampa bay buccaneers": "TB", "buccaneers": "TB", "bucs": "TB", "tampa bay": "TB", "tb": "TB",
    # Titans
    "tennessee titans": "TEN", "titans": "TEN", "tennessee": "TEN", "ten": "TEN",
    # Commanders (incl. legacy)
    "washington commanders": "WAS", "commanders": "WAS", "washington": "WAS", "was": "WAS", "wsh": "WAS",
    "washington football team": "WAS", "redskins": "WAS"
}

CODE_TO_FULLNAME = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams", "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders"
}

def team_key(name: str) -> str:
    """Map any team string (full, nickname, abbreviation) to a canonical 2–3 letter code."""
    if pd.isna(name):
        return ""
    s = str(name).strip().lower()
    return TEAM_ALIAS_TO_CODE.get(s, s)  # fall back to cleaned string if unknown

# =========================
# Sidebar: Cache refresh
# =========================
with st.sidebar:
    st.header("Data")
    if st.button("🔄 Clear cache & reload data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# =========================
# Helpers
# =========================
def normalize_header(name: str) -> str:
    name = str(name) if not isinstance(name, str) else name
    name = name.strip().replace(" ", "_").lower()
    name = re.sub(r"[^0-9a-z_]", "", name)
    return name

@st.cache_data(show_spinner=False)
def load_scores() -> pd.DataFrame:
    df = pd.read_csv(SCORE_URL)
    df.columns = [normalize_header(c) for c in df.columns]
    # Add canonical keys for join logic
    if "home_team" in df.columns:
        df["home_key"] = df["home_team"].apply(team_key)
    if "away_team" in df.columns:
        df["away_key"] = df["away_team"].apply(team_key)
    return df

def load_and_clean(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = [normalize_header(c) for c in df.columns]
    # Ensure a 'team' column and compute team_key
    if "team" in df.columns:
        df["team"] = df["team"].astype(str).str.strip()
    elif "teams" in df.columns:
        df["team"] = df["teams"].astype(str).str.strip()
    else:
        # create empty to avoid key errors later
        df["team"] = ""
    df["team_key"] = df["team"].apply(team_key)
    return df

@st.cache_data(show_spinner=False)
def load_all_player_dfs():
    return {name: load_and_clean(url) for name, url in SHEETS.items()}

def avg_scoring(df: pd.DataFrame, team_label: str):
    """Keeps your original logic using the labels present in the SCORE sheet."""
    scored_home = df.loc[df["home_team"] == team_label, "home_score"].mean()
    scored_away = df.loc[df["away_team"] == team_label, "away_score"].mean()
    allowed_home = df.loc[df["home_team"] == team_label, "away_score"].mean()
    allowed_away = df.loc[df["away_team"] == team_label, "home_score"].mean()
    return np.nanmean([scored_home, scored_away]), np.nanmean([allowed_home, allowed_away])

def predict_scores(df: pd.DataFrame, team_label: str, opponent_label: str):
    team_avg_scored, team_avg_allowed = avg_scoring(df, team_label)
    opp_avg_scored, opp_avg_allowed = avg_scoring(df, opponent_label)

    raw_team_pts = (team_avg_scored + opp_avg_allowed) / 2
    raw_opp_pts = (opp_avg_scored + team_avg_allowed) / 2

    league_avg_pts = df[["home_score", "away_score"]].stack().mean()
    cal_factor = 22.3 / league_avg_pts if not np.isnan(league_avg_pts) and league_avg_pts > 0 else 1.0

    team_pts = float(raw_team_pts * cal_factor) if pd.notna(raw_team_pts) else 22.3
    opp_pts = float(raw_opp_pts * cal_factor) if pd.notna(raw_opp_pts) else 22.3
    return team_pts, opp_pts

# ===== Prop helpers =====
def find_player_in(df: pd.DataFrame, player_name: str):
    if "player" not in df.columns:
        return None
    mask = df["player"].astype(str).str.lower() == str(player_name).lower()
    return df[mask].copy() if mask.any() else None

def detect_stat_col(df: pd.DataFrame, prop: str):
    cols = list(df.columns)
    norm = [normalize_header(c) for c in cols]
    mapping = {
        "rushing_yards": ["rushing_yards_total", "rushing_yards_per_game"],
        "receiving_yards": ["receiving_yards_total", "receiving_yards_per_game"],
        "passing_yards": ["passing_yards_total", "passing_yards_per_game"],
        "receptions": ["receiving_receptions_total"],
        "targets": ["receiving_targets_total"],
        "carries": ["rushing_attempts_total", "rushing_carries_per_game"],
    }
    pri = mapping.get(prop, [])
    for cand in pri:
        if cand in norm:
            return cols[norm.index(cand)]
    return None

def pick_def_df(prop: str, pos: str, d_qb, d_rb, d_wr, d_te):
    if prop == "passing_yards":
        return d_qb
    if prop in ["rushing_yards", "carries"]:
        return d_rb if pos != "qb" else d_qb
    if prop in ["receiving_yards", "receptions", "targets"]:
        if pos == "te":
            return d_te
        if pos == "rb":
            return d_rb
        return d_wr
    return None

def detect_def_col(def_df: pd.DataFrame, prop: str):
    cols = list(def_df.columns)
    norm = [normalize_header(c) for c in cols]
    prefs = []
    if prop in ["rushing_yards", "carries"]:
        prefs = ["rushing_yards_allowed_total", "rushing_yards_allowed"]
    elif prop in ["receiving_yards", "receptions", "targets"]:
        prefs = ["receiving_yards_allowed_total", "receiving_yards_allowed"]
    elif prop == "passing_yards":
        prefs = ["passing_yards_allowed_total", "passing_yards_allowed"]
    for cand in prefs:
        if cand in norm:
            return cols[norm.index(cand)]
    for i, nc in enumerate(norm):
        if "allowed" in nc:
            return cols[i]
    return None

# =========================
# UI – Single Page
# =========================
st.title("🏈 NFL Game + Player Prop Dashboard (One-Page)")

scores_df = load_scores()
if scores_df.empty:
    st.error("Could not load NFL game data.")
    st.stop()

player_data = load_all_player_dfs()
p_rec, p_rush, p_pass = player_data["player_receiving"], player_data["player_rushing"], player_data["player_passing"]
d_rb, d_qb, d_wr, d_te = player_data["def_rb"], player_data["def_qb"], player_data["def_wr"], player_data["def_te"]

# -------------------------
# 1) Week & Team selection
# -------------------------
with st.container():
    st.header("1) Select Game")
    cols = st.columns([1, 1, 2])
    with cols[0]:
        week_list = sorted(scores_df["week"].dropna().unique())
        selected_week = st.selectbox("Week", week_list)
    with cols[1]:
        teams_in_week = sorted(
            set(scores_df.loc[scores_df["week"] == selected_week, "home_team"].dropna().unique())
            | set(scores_df.loc[scores_df["week"] == selected_week, "away_team"].dropna().unique())
        )
        selected_team = st.selectbox("Team", teams_in_week)

    # Find game row & opponent using labels from SCORE sheet
    game_row = scores_df[
        ((scores_df["home_team"] == selected_team) | (scores_df["away_team"] == selected_team))
        & (scores_df["week"] == selected_week)
    ]
    if game_row.empty:
        st.warning("No game found for that team/week.")
        st.stop()
    g = game_row.iloc[0]
    opponent = g["away_team"] if g["home_team"] == selected_team else g["home_team"]

    # Canonical keys for cross-sheet joins
    selected_team_key = team_key(selected_team)
    opponent_key = team_key(opponent)

    with cols[2]:
        st.markdown(f"**Matchup:** {selected_team} vs {opponent}")

    # Lines (pre-fill from sheet if present)
    default_ou = float(g.get("over_under", 45.0)) if pd.notna(g.get("over_under", np.nan)) else 45.0
    default_spread = float(g.get("spread", 0.0)) if pd.notna(g.get("spread", np.nan)) else 0.0

    cL, cR = st.columns(2)
    with cL:
        over_under = st.number_input("Over/Under (Vegas or yours)", value=default_ou, step=0.5)
    with cR:
        spread = st.number_input("Spread (negative = favorite)", value=default_spread, step=0.5)

# -------------------------
# 2) Game prediction (Vegas-calibrated)
# -------------------------
with st.container():
    st.header("2) Game Prediction (Vegas-Calibrated)")
    team_pts, opp_pts = predict_scores(scores_df, selected_team, opponent)
    total_pred = team_pts + opp_pts
    margin = team_pts - opp_pts
    total_diff = total_pred - over_under
    spread_diff = margin - (-spread)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{selected_team} Predicted", f"{team_pts:.1f} pts")
    m2.metric(f"{opponent} Predicted", f"{opp_pts:.1f} pts")
    m3.metric("Predicted Total", f"{total_pred:.1f}", f"{total_diff:+.1f} vs O/U")
    m4.metric("Predicted Margin", f"{margin:+.1f}", f"{spread_diff:+.1f} vs Spread")

    fig_total = px.bar(
        x=["Predicted Total", "Vegas O/U"],
        y=[total_pred, over_under],
        title="Predicted Total vs O/U"
    )
    st.plotly_chart(fig_total, use_container_width=True)

    fig_margin = px.bar(
        x=["Predicted Margin", "Vegas Spread"],
        y=[margin, -spread],
        title="Predicted Margin vs Spread"
    )
    st.plotly_chart(fig_margin, use_container_width=True)

# -------------------------
# 3) Top Edges of the Week
# -------------------------
with st.container():
    st.header("3) Top Edges This Week")
    wk = scores_df[scores_df["week"] == selected_week].copy()
    rows = []
    for _, r in wk.iterrows():
        h, a = r.get("home_team"), r.get("away_team")
        if pd.isna(h) or pd.isna(a):
            continue
        h_pts, a_pts = predict_scores(scores_df, h, a)
        tot = h_pts + a_pts
        mar = h_pts - a_pts
        ou = float(r.get("over_under")) if pd.notna(r.get("over_under", np.nan)) else np.nan
        sp = float(r.get("spread")) if pd.notna(r.get("spread", np.nan)) else np.nan
        total_edge = np.nan if pd.isna(ou) else tot - ou
        spread_edge = np.nan if pd.isna(sp) else mar - (-sp)
        rows.append({
            "Matchup": f"{a} @ {h}",
            "Pred Total": round(tot, 1),
            "O/U": ou if not pd.isna(ou) else "",
            "Total Edge (pts)": None if pd.isna(total_edge) else round(total_edge, 1),
            "Pred Margin": round(mar, 1),
            "Spread": sp if not pd.isna(sp) else "",
            "Spread Edge (pts)": None if pd.isna(spread_edge) else round(spread_edge, 1),
        })
    edges_df = pd.DataFrame(rows)
    if not edges_df.empty:
        # Rank by absolute edge (take the larger of total/spread edge per row)
        def edge_rank(row):
            vals = [abs(v) for v in [row.get("Total Edge (pts)"), row.get("Spread Edge (pts)")] if pd.notna(v)]
            return max(vals) if vals else 0.0
        edges_df["Abs Edge"] = edges_df.apply(edge_rank, axis=1)
        edges_df = edges_df.sort_values("Abs Edge", ascending=False).drop(columns=["Abs Edge"])
        st.dataframe(edges_df, use_container_width=True)
    else:
        st.info("No games found for this week.")

# -------------------------
# 4) Player Props (players from both teams)
# -------------------------
with st.container():
    st.header("4) Player Props (Both Teams)")

    # Build player list from both teams using canonical keys
    def players_for_team(df, team_name_or_label):
        key = team_key(team_name_or_label)
        if "team_key" not in df.columns or "player" not in df.columns:
            return []
        mask = df["team_key"] == key
        return list(df.loc[mask, "player"].dropna().unique())

    team_players = set(
        players_for_team(p_rec, selected_team) +
        players_for_team(p_rush, selected_team) +
        players_for_team(p_pass, selected_team)
    )
    opp_players = set(
        players_for_team(p_rec, opponent) +
        players_for_team(p_rush, opponent) +
        players_for_team(p_pass, opponent)
    )
    both_players = sorted(team_players.union(opp_players))

    # Helpful message if no players found (usually a team alias mismatch)
    if not both_players:
        st.info(
            "No players found for this matchup. This often means team labels differ across sheets. "
            f"Resolved keys — Your selection: **{selected_team_key}**, Opponent: **{opponent_key}**."
        )

    c1, c2, c3 = st.columns([2, 1.2, 1.2])
    with c1:
        player_name = st.selectbox("Select Player", [""] + both_players)
    with c2:
        prop_choices = ["passing_yards", "rushing_yards", "receiving_yards", "receptions", "targets", "carries", "anytime_td"]
        selected_prop = st.selectbox("Prop Type", prop_choices, index=2)
    with c3:
        default_line = 50.0 if selected_prop != "anytime_td" else 0.0
        line_val = st.number_input("Sportsbook Line", value=float(default_line)) if selected_prop != "anytime_td" else 0.0

    if player_name:
        # Determine which sheet to use for the player
        def pick_player_df(prop):
            if prop in ["receiving_yards", "receptions", "targets"]:
                return p_rec, "wr"
            if prop in ["rushing_yards", "carries"]:
                return p_rush, "rb"
            if prop == "passing_yards":
                return p_pass, "qb"
            # Fallback for anytime TD: search both rec + rush
            return p_rec, "wr"

        player_df_source, fallback_pos = pick_player_df(selected_prop)
        this_player_df = find_player_in(player_df_source, player_name)

        # --- ANYTIME TD: compute even if player is found in one table (we use rush+rec totals)
        if selected_prop == "anytime_td":
            rec_row = find_player_in(p_rec, player_name)
            rush_row = find_player_in(p_rush, player_name)

            total_tds = 0.0
            total_games = 0.0
            for df_ in [rec_row, rush_row]:
                if df_ is not None and not df_.empty:
                    td_cols = [c for c in df_.columns if "td" in c and "allowed" not in c]
                    games_col = "games_played" if "games_played" in df_.columns else None
                    if td_cols and games_col:
                        tds = sum(float(df_.iloc[0][col]) for col in td_cols if pd.notna(df_.iloc[0][col]))
                        total_tds += tds
                        total_games = max(total_games, float(df_.iloc[0][games_col]))
            if total_games == 0:
                st.warning("No touchdown data found for this player.")
            else:
                # Defense context (RB/WR/TE) — all get a TDs/game metric
                def_dfs = [d_rb.copy(), d_wr.copy(), d_te.copy()]
                for d in def_dfs:
                    if "games_played" not in d.columns:
                        d["games_played"] = 1
                    td_cols = [c for c in d.columns if "td" in c and "allowed" in c]
                    d["tds_pg"] = d[td_cols].sum(axis=1) / d["games_played"].replace(0, np.nan)
                    if "team_key" not in d.columns:
                        d["team_key"] = d["team"].apply(team_key)

                league_td_pg = np.nanmean([d["tds_pg"].mean() for d in def_dfs])

                # Determine player's team key → opponent key
                player_team_key = None
                for df_ in [p_rec, p_rush, p_pass]:
                    row_ = find_player_in(df_, player_name)
                    if row_ is not None and not row_.empty:
                        tk = row_.iloc[0].get("team_key", "")
                        if tk:
                            player_team_key = tk
                            break
                if not player_team_key:
                    # Fallback: assume on selected side
                    player_team_key = selected_team_key

                opp_key_for_player = opponent_key if player_team_key == selected_team_key else selected_team_key

                opp_td_list = []
                for d in def_dfs:
                    mask = d["team_key"] == opp_key_for_player
                    opp_td_list.append(d.loc[mask, "tds_pg"].mean())
                opp_td_pg = np.nanmean(opp_td_list)
                if np.isnan(opp_td_pg):
                    opp_td_pg = league_td_pg

                adj_factor = (opp_td_pg / league_td_pg) if league_td_pg and league_td_pg > 0 else 1.0
                adj_td_rate = (total_tds / total_games) * adj_factor
                # Convert rate to probability (Poisson ~ at least one TD)
                prob_anytime = 1 - np.exp(-adj_td_rate)
                prob_anytime = float(np.clip(prob_anytime, 0.0, 1.0))

                opp_label = CODE_TO_FULLNAME.get(opp_key_for_player, opp_key_for_player)
                st.subheader("Anytime TD Probability")
                st.write(f"Estimated Anytime TD Probability: **{prob_anytime*100:.1f}%**")

                bar_df = pd.DataFrame(
                    {"Category": ["Player TDs/Game", "Adj. vs Opponent"], "TDs/Game": [(total_tds/total_games), adj_td_rate]}
                )
                st.plotly_chart(
                    px.bar(bar_df, x="Category", y="TDs/Game", title=f"{player_name} – Anytime TD vs {opp_label}"),
                    use_container_width=True
                )

        # --- Non-TD props
        elif this_player_df is None or this_player_df.empty:
            st.warning("Player not found in the selected stat table.")
        else:
            player_pos = this_player_df.iloc[0].get("position", fallback_pos)
            stat_col = detect_stat_col(this_player_df, selected_prop)
            if not stat_col:
                st.warning("No matching stat column found for this prop.")
            else:
                # Season per-game from player table
                season_val = float(this_player_df.iloc[0][stat_col]) if pd.notna(this_player_df.iloc[0][stat_col]) else 0.0
                games_played = float(this_player_df.iloc[0].get("games_played", 1)) or 1.0
                player_pg = season_val / games_played if games_played > 0 else 0.0

                # Defense adjust
                def_df = pick_def_df(selected_prop, player_pos, d_qb, d_rb, d_wr, d_te)
                def_col = detect_def_col(def_df, selected_prop) if def_df is not None else None

                # Figure opponent for THIS player via team keys
                player_team_key = str(this_player_df.iloc[0].get("team_key", "")).strip()
                opp_key_for_player = opponent_key if player_team_key == selected_team_key else selected_team_key

                opp_allowed_pg = None
                league_allowed_pg = None

                if def_df is not None and def_col is not None:
                    if "team_key" not in def_df.columns:
                        def_df["team_key"] = def_df["team"].apply(team_key)

                    if "games_played" in def_df.columns:
                        league_allowed_pg = (def_df[def_col] / def_df["games_played"].replace(0, np.nan)).mean()
                    else:
                        league_allowed_pg = def_df[def_col].mean()

                    opp_row = def_df[def_df["team_key"] == opp_key_for_player]
                    if not opp_row.empty:
                        if "games_played" in opp_row.columns and float(opp_row.iloc[0]["games_played"]) > 0:
                            opp_allowed_pg = float(opp_row.iloc[0][def_col]) / float(opp_row.iloc[0]["games_played"])
                        else:
                            opp_allowed_pg = float(opp_row.iloc[0][def_col])
                    else:
                        opp_allowed_pg = league_allowed_pg

                adj_factor = (opp_allowed_pg / league_allowed_pg) if (league_allowed_pg and league_allowed_pg > 0) else 1.0
                predicted_pg = player_pg * adj_factor

                stdev = max(3.0, predicted_pg * 0.35)
                z = (line_val - predicted_pg) / stdev
                prob_over = float(np.clip(1 - norm.cdf(z), 0.001, 0.999))
                prob_under = float(np.clip(norm.cdf(z), 0.001, 0.999))

                st.subheader(selected_prop.replace("_", " ").title())
                st.write(f"**Player (season total):** {season_val:.2f} over {games_played:.0f} games → **{player_pg:.2f} per game**")
                st.write(f"**Adjusted prediction (this game):** {predicted_pg:.2f}")
                st.write(f"**Line:** {line_val:.1f}")
                st.write(f"**Probability of OVER:** {prob_over*100:.1f}%")
                st.write(f"**Probability of UNDER:** {prob_under*100:.1f}%")

                st.plotly_chart(
                    px.bar(
                        x=["Predicted (this game)", "Line"],
                        y=[predicted_pg, line_val],
                        title=f"{player_name} – {selected_prop.replace('_', ' ').title()}"
                    ),
                    use_container_width=True
                )
    else:
        st.info("Select a player to evaluate props.")

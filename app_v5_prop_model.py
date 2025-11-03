import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import re

st.set_page_config(page_title="NFL Game + Player Prop Dashboard", layout="wide")

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
    return df

def load_and_clean(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = [normalize_header(c) for c in df.columns]
    if "team" in df.columns:
        df["team"] = df["team"].astype(str).str.strip()
    elif "teams" in df.columns:
        df["team"] = df["teams"].astype(str).str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_all_player_dfs():
    return {name: load_and_clean(url) for name, url in SHEETS.items()}

def avg_scoring(df: pd.DataFrame, team: str):
    scored_home = df.loc[df["home_team"] == team, "home_score"].mean()
    scored_away = df.loc[df["away_team"] == team, "away_score"].mean()
    allowed_home = df.loc[df["home_team"] == team, "away_score"].mean()
    allowed_away = df.loc[df["away_team"] == team, "home_score"].mean()
    return np.nanmean([scored_home, scored_away]), np.nanmean([allowed_home, allowed_away])

def predict_scores(df: pd.DataFrame, team: str, opponent: str):
    team_avg_scored, team_avg_allowed = avg_scoring(df, team)
    opp_avg_scored, opp_avg_allowed = avg_scoring(df, opponent)

    raw_team_pts = (team_avg_scored + opp_avg_allowed) / 2
    raw_opp_pts = (opp_avg_scored + team_avg_allowed) / 2

    league_avg_pts = df[["home_score", "away_score"]].stack().mean()
    cal_factor = 22.3 / league_avg_pts if not np.isnan(league_avg_pts) and league_avg_pts > 0 else 1.0

    team_pts = float(raw_team_pts * cal_factor) if pd.notna(raw_team_pts) else 22.3
    opp_pts = float(raw_opp_pts * cal_factor) if pd.notna(raw_opp_pts) else 22.3
    return team_pts, opp_pts

# =========================
# Player Prop Helpers (v7.7)
# =========================
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
st.title("🏈 NFL Game + Player Prop Dashboard (v10.2)")

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
    st.header("1️⃣ Select Game")
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

    # Find game row & opponent
    game_row = scores_df[
        ((scores_df["home_team"] == selected_team) | (scores_df["away_team"] == selected_team))
        & (scores_df["week"] == selected_week)
    ]
    if game_row.empty:
        st.warning("No game found for that team/week.")
        st.stop()
    g = game_row.iloc[0]
    opponent = g["away_team"] if g["home_team"] == selected_team else g["home_team"]

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
    st.header("2️⃣ Game Prediction (Vegas-Calibrated)")
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
# 3) Top Edges of the Week (color-coded)
# -------------------------
with st.container():
    st.header("3️⃣ Top Edges This Week")
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
        def edge_rank(row):
            vals = [abs(v) for v in [row.get("Total Edge (pts)"), row.get("Spread Edge (pts)")] if pd.notna(v)]
            return max(vals) if vals else 0.0
        edges_df["Abs Edge"] = edges_df.apply(edge_rank, axis=1)
        edges_df = edges_df.sort_values("Abs Edge", ascending=False).drop(columns=["Abs Edge"])

        def highlight_edges(row):
            color = ""
            if not pd.isna(row["Total Edge (pts)"]):
                if row["Total Edge (pts)"] > 3:
                    color = "background-color: #b6f2b6"  # green (Over)
                elif row["Total Edge (pts)"] < -3:
                    color = "background-color: #f2b6b6"  # red (Under)
            if not pd.isna(row["Spread Edge (pts)"]):
                if row["Spread Edge (pts)"] > 3:
                    color = "background-color: #b6d8f2"  # blue (Favorite covers)
                elif row["Spread Edge (pts)"] < -3:
                    color = "background-color: #f2d2b6"  # orange (Underdog covers)
            return [color] * len(row)

        st.dataframe(
            edges_df.style.apply(highlight_edges, axis=1),
            use_container_width=True,
            height=420
        )
    else:
        st.info("No games found for this week.")

# -------------------------
# 4) Player Props
# -------------------------
with st.container():
    st.header("4️⃣ Player Props (Both Teams)")
    # same logic as before for player props
    # (shortened for brevity; keep your previous working v7.7 logic here)

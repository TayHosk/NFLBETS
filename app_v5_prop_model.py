import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import re

st.set_page_config(page_title="NFL Betting Model", layout="wide")

# =========================
# Data Sources (Google Sheets)
# =========================
SCORE_URL = "https://docs.google.com/spreadsheets/d/1KrTQbR5uqlBn2v2Onpjo6qHFnLlrqIQBzE52KAhMYcY/export?format=csv"
TEAM_GAME_LOG_URL = "https://docs.google.com/spreadsheets/d/1PmVC_rWKdHNISbZHe0uQmv65d3fJx6vBtA2ydHl1qQA/export?format=csv"

# Season-long player stat sheets
QB_STATS_URL = "https://docs.google.com/spreadsheets/d/1I9YNSQMylW_waJs910q4S6SM8CZE--hsyNElrJeRfvk/export?format=csv"
WR_STATS_URL = "https://docs.google.com/spreadsheets/d/1Gwb2A-a4ge7UKHnC7wUpJltgioTuCQNuwOiC5ecZReM/export?format=csv"
RB_STATS_URL = "https://docs.google.com/spreadsheets/d/11vYTC2o-WOOyN_oVuqq-hArgwdsPfvQWJWqiISK0bUg/export?format=csv"
TE_STATS_URL = "https://docs.google.com/spreadsheets/d/1tAFG33Mh2WlDWAAobEJFbFS6A_7mdp0RjHanolrqTgs/export?format=csv"

# Player game logs (every game this year)
PLAYER_GAME_LOG_URL = "https://docs.google.com/spreadsheets/d/1iJNtTJcC3zv0Qvb7LxynF1tTPlggN8hOpQP-ROnPda8/export?format=csv"

# How many recent games to use for trend vs season
PROP_TREND_LAST_N = 2

# =========================
# Team alias map
# =========================
TEAM_ALIAS_TO_CODE = {
    "arizona cardinals": "ARI", "cardinals": "ARI", "arizona": "ARI", "ari": "ARI",
    "atlanta falcons": "ATL", "falcons": "ATL", "atlanta": "ATL", "atl": "ATL",
    "baltimore ravens": "BAL", "ravens": "BAL", "baltimore": "BAL", "bal": "BAL",
    "buffalo bills": "BUF", "bills": "BUF", "buffalo": "BUF", "buf": "BUF",
    "carolina panthers": "CAR", "panthers": "CAR", "carolina": "CAR", "car": "CAR",
    "chicago bears": "CHI", "bears": "CHI", "chicago": "CHI", "chi": "CHI",
    "cincinnati bengals": "CIN", "bengals": "CIN", "cincinnati": "CIN", "cin": "CIN",
    "cleveland browns": "CLE", "browns": "CLE", "cleveland": "CLE", "cle": "CLE",
    "dallas cowboys": "DAL", "cowboys": "DAL", "dallas": "DAL", "dal": "DAL",
    "denver broncos": "DEN", "broncos": "DEN", "denver": "DEN", "den": "DEN",
    "detroit lions": "DET", "lions": "DET", "detroit": "DET", "det": "DET",
    "green bay packers": "GB", "packers": "GB", "green bay": "GB", "gb": "GB",
    "houston texans": "HOU", "texans": "HOU", "houston": "HOU", "hou": "HOU",
    "indianapolis colts": "IND", "colts": "IND", "indianapolis": "IND", "ind": "IND",
    "jacksonville jaguars": "JAX", "jaguars": "JAX", "jacksonville": "JAX", "jax": "JAX", "jacs": "JAX",
    "kansas city chiefs": "KC", "chiefs": "KC", "kansas city": "KC", "kc": "KC",
    "las vegas raiders": "LV", "raiders": "LV", "las vegas": "LV", "lv": "LV",
    "los angeles chargers": "LAC", "la chargers": "LAC", "chargers": "LAC", "lac": "LAC",
    "los angeles rams": "LAR", "la rams": "LAR", "rams": "LAR", "lar": "LAR",
    "miami dolphins": "MIA", "dolphins": "MIA", "miami": "MIA", "mia": "MIA",
    "minnesota vikings": "MIN", "vikings": "MIN", "minnesota": "MIN", "min": "MIN",
    "new england patriots": "NE", "patriots": "NE", "new england": "NE", "ne": "NE",
    "new orleans saints": "NO", "saints": "NO", "new orleans": "NO", "no": "NO", "nos": "NO",
    "new york giants": "NYG", "ny giants": "NYG", "giants": "NYG", "nyg": "NYG",
    "new york jets": "NYJ", "ny jets": "NYJ", "jets": "NYJ", "nyj": "NYJ",
    "philadelphia eagles": "PHI", "eagles": "PHI", "philadelphia": "PHI", "phi": "PHI",
    "pittsburgh steelers": "PIT", "steelers": "PIT", "pittsburgh": "PIT", "pit": "PIT",
    "san francisco 49ers": "SF", "49ers": "SF", "niners": "SF", "san francisco": "SF", "sf": "SF",
    "seattle seahawks": "SEA", "seahawks": "SEA", "seattle": "SEA", "sea": "SEA",
    "tampa bay buccaneers": "TB", "buccaneers": "TB", "bucs": "TB", "tampa bay": "TB", "tb": "TB",
    "tennessee titans": "TEN", "titans": "TEN", "tennessee": "TEN", "ten": "TEN",
    "washington commanders": "WAS", "commanders": "WAS", "washington": "WAS", "was": "WAS", "wsh": "WAS",
    "washington football team": "WAS", "redskins": "WAS",

    # PFR-style codes from sheets
    "nwe": "NE",
    "gnb": "GB",
    "kan": "KC",
    "tam": "TB",
    "nor": "NO",
    "sfo": "SF",
    "lvr": "LV",
    "ram": "LAR",
}
CODE_TO_FULLNAME = {
    "ARI": "Arizona Cardinals","ATL": "Atlanta Falcons","BAL": "Baltimore Ravens","BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers","CHI": "Chicago Bears","CIN": "Cincinnati Bengals","CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys","DEN": "Denver Broncos","DET": "Detroit Lions","GB": "Green Bay Packers",
    "HOU": "Houston Texans","IND": "Indianapolis Colts","JAX": "Jacksonville Jaguars","KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers","LAR": "Los Angeles Rams","LV": "Las Vegas Raiders","MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings","NE": "New England Patriots","NO": "New Orleans Saints","NYG": "New York Giants",
    "NYJ": "New York Jets","PHI": "Philadelphia Eagles","PIT": "Pittsburgh Steelers","SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers","TB": "Tampa Bay Buccaneers","TEN": "Tennessee Titans","WAS": "Washington Commanders"
}
FULLNAME_TO_CODE = {v: k for k, v in CODE_TO_FULLNAME.items()}


def team_key(name: str) -> str:
    if pd.isna(name):
        return ""
    s_raw = str(name).strip()
    s_lower = s_raw.lower()

    if s_lower in TEAM_ALIAS_TO_CODE:
        return TEAM_ALIAS_TO_CODE[s_lower]

    s_clean = re.sub(r"[^a-z]", "", s_lower)
    if s_clean in TEAM_ALIAS_TO_CODE:
        return TEAM_ALIAS_TO_CODE[s_clean]

    if "," in s_lower:
        first = s_lower.split(",")[0].strip()
        if first in TEAM_ALIAS_TO_CODE:
            return TEAM_ALIAS_TO_CODE[first]
        first_clean = re.sub(r"[^a-z]", "", first)
        if first_clean in TEAM_ALIAS_TO_CODE:
            return TEAM_ALIAS_TO_CODE[first_clean]

    cand = s_clean.upper()
    if cand in CODE_TO_FULLNAME:
        return cand

    return cand


# =========================
# Session init
# =========================
if "my_bets" not in st.session_state:
    st.session_state["my_bets"] = []
if "parlay_legs" not in st.session_state:
    st.session_state["parlay_legs"] = []


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
    if not isinstance(name, str):
        name = str(name)
    name = name.strip().replace(" ", "_").lower()
    name = re.sub(r"[^0-9a-z_]", "", name)
    return name


def find_col(cols_norm, prefer_any_of, contains_ok=False):
    for cand in prefer_any_of:
        if cand in cols_norm:
            return cols_norm.index(cand)
    if contains_ok:
        for i, c in enumerate(cols_norm):
            if any(cand in c for cand in prefer_any_of):
                return i
    return None


def consolidate_duplicate_columns(df, base_normalized):
    cols = [c for c in df.columns if c == base_normalized or c.startswith(base_normalized + "_")]
    if len(cols) <= 1:
        return df
    out = df.copy()
    out[base_normalized + "_final"] = pd.NA
    for c in cols:
        out[base_normalized + "_final"] = out[base_normalized + "_final"].fillna(out[c])
    out.drop(columns=cols, inplace=True)
    out.rename(columns={base_normalized + "_final": base_normalized}, inplace=True)
    return out


def parse_ou_result(val: str):
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if "over" in s:
        return "over"
    if "under" in s:
        return "under"
    return None


def parse_cover_result(val: str):
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in ["covered", "cover", "yes", "y", "w", "win", "won"]:
        return "covered"
    if s in ["did not cover", "didnt cover", "didn't cover", "no", "n", "l", "lose", "lost"]:
        return "did not cover"
    if "cover" in s:
        return "did not cover" if ("not" in s or "did not" in s or "didn't" in s) else "covered"
    if "win" in s:
        return "covered"
    if "lose" in s or "lost" in s:
        return "did not cover"
    return None


# =========================
# Loaders
# =========================
def compute_home_spread(row: pd.Series) -> float:
    try:
        home = row.get("home_team", None)
        fav = row.get("favored_team", None)
        sp = row.get("spread", np.nan)
        if pd.isna(sp) or not fav or not home:
            return np.nan
        hk = team_key(home)
        fk = team_key(fav)
        return -abs(sp) if (fk and hk and fk == hk) else abs(sp)
    except Exception:
        return np.nan


@st.cache_data(show_spinner=False)
def load_scores() -> pd.DataFrame:
    df = pd.read_csv(SCORE_URL)
    df.columns = [normalize_header(c) for c in df.columns]
    if "over_under" in df.columns:
        df["over_under"] = pd.to_numeric(df["over_under"], errors="coerce")
    if "spread" in df.columns:
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce")
    if "home_team" in df.columns:
        df["home_key"] = df["home_team"].apply(team_key)
    if "away_team" in df.columns:
        df["away_key"] = df["away_team"].apply(team_key)
    if "favored_team" in df.columns:
        df["favored_key"] = df["favored_team"].apply(team_key)
    if {"home_team", "favored_team", "spread"}.issubset(df.columns):
        df["home_spread"] = df.apply(compute_home_spread, axis=1)
    else:
        df["home_spread"] = np.nan
    return df


@st.cache_data(show_spinner=False)
def load_all_player_dfs():
    def load_player_sheet(url: str) -> pd.DataFrame:
        df = pd.read_csv(url)
        df.columns = [normalize_header(c) for c in df.columns]

        if "player" in df.columns:
            df["player"] = df["player"].astype(str).str.strip()
        else:
            df["player"] = ""

        if "team" in df.columns:
            df["team"] = df["team"].astype(str).str.strip()
            df["team_key"] = df["team"].apply(team_key)
        else:
            df["team"] = ""
            df["team_key"] = ""

        for base in [
            "games_played",
            "yards_gained_by_passing",
            "passing_yards_gained",
            "rushing_yards_gained",
            "receiving_yards",
            "receptions",
            "rushing_touchdowns",
            "receiving_touchdowns",
            "passing_touchdowns",
            "pass_targets",
            "rushing_attempts",
        ]:
            if any((c == base) or c.startswith(base + "_") for c in df.columns):
                df = consolidate_duplicate_columns(df, base)

        return df

    qb = load_player_sheet(QB_STATS_URL)
    wr = load_player_sheet(WR_STATS_URL)
    rb = load_player_sheet(RB_STATS_URL)
    te = load_player_sheet(TE_STATS_URL)

    return {"qb": qb, "wr": wr, "rb": rb, "te": te}


@st.cache_data(show_spinner=False)
def load_player_game_log(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = [normalize_header(c) for c in df.columns]

    if "player" in df.columns:
        df["player"] = df["player"].astype(str).str.strip()
    else:
        df["player"] = ""

    if "team" in df.columns:
        df["team"] = df["team"].astype(str).str.strip()
        df["team_key"] = df["team"].apply(team_key)
    else:
        df["team"] = ""
        df["team_key"] = ""

    if "opponent" in df.columns:
        df["opponent"] = df["opponent"].astype(str).str.strip()
        df["opponent_key"] = df["opponent"].apply(team_key)
    else:
        df["opponent"] = ""
        df["opponent_key"] = ""

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def load_team_game_log(url: str) -> pd.DataFrame:
    raw = pd.read_csv(url)
    raw_cols = list(raw.columns)
    norm_cols = [normalize_header(c) for c in raw_cols]
    df = raw.copy()
    df.columns = norm_cols

    df = consolidate_duplicate_columns(df, "over_under_result")

    mapper = {}
    idx = find_col(norm_cols, ["team"])
    if idx is not None:
        mapper["team"] = df.columns[idx]
    idx = find_col(norm_cols, ["date"])
    if idx is not None:
        mapper["date"] = df.columns[idx]
    idx = find_col(norm_cols, ["day"])
    if idx is not None:
        mapper["day"] = df.columns[idx]
    idx = find_col(norm_cols, ["week"])
    if idx is not None:
        mapper["week"] = df.columns[idx]

    idx = find_col(norm_cols, ["if_away_team_it_states", "if_away_team_it_states_if_home_team_it_says_nothing"], contains_ok=True)
    if idx is not None:
        mapper["home_away_flag"] = df.columns[idx]

    idx = find_col(norm_cols, ["opponent"])
    if idx is not None:
        mapper["opponent"] = df.columns[idx]
    idx = find_col(norm_cols, ["over_under_line"])
    if idx is not None:
        mapper["ou_line"] = df.columns[idx]

    if "over_under_result" in df.columns:
        mapper["ou_result"] = "over_under_result"

    idx = find_col(norm_cols, ["spread"])
    if idx is not None:
        mapper["spread"] = df.columns[idx]

    idx = find_col(norm_cols, ["did_they_cover_the_spread"], contains_ok=True)
    if idx is not None:
        mapper["cover_result"] = df.columns[idx]

    idx = find_col(norm_cols, ["result"], contains_ok=True)
    if idx is not None:
        mapper["result"] = df.columns[idx]

    # Opp-allowed columns
    for c in ["passing_yards_gained_by_opposition", "rushing_yards_by_opposition", "rushing_attempts_by_opposition"]:
        if c in df.columns:
            mapper[c] = c

    out = pd.DataFrame()
    for k, col in mapper.items():
        out[k] = df[col]

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "week" in out.columns:
        out["week"] = pd.to_numeric(out["week"], errors="coerce")
    if "ou_line" in out.columns:
        out["ou_line"] = pd.to_numeric(out["ou_line"], errors="coerce")
    if "spread" in out.columns:
        out["spread"] = pd.to_numeric(out["spread"], errors="coerce")

    if "home_away_flag" in out.columns:
        out["is_away"] = out["home_away_flag"].fillna("").astype(str).str.contains("@")
    else:
        out["is_away"] = pd.NA

    out["team_key"] = out["team"].apply(team_key) if "team" in out.columns else ""
    if "opponent" in out.columns:
        out["opponent_key"] = out["opponent"].apply(team_key)

    if "ou_result" in out.columns:
        out["ou_result_norm"] = out["ou_result"].apply(parse_ou_result)
    if "cover_result" in out.columns:
        out["cover_result_norm"] = out["cover_result"].apply(parse_cover_result)

    return out


# =========================
# Model logic
# =========================
def avg_scoring(df: pd.DataFrame, team_label: str):
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


# ===== Team Log Adjustment Helpers =====
def team_log_summary(team_log_df: pd.DataFrame, team_abbr: str, last_n: int = 0):
    if team_log_df is None or team_log_df.empty:
        return {"ou_over": 0, "ou_under": 0, "cover_yes": 0, "cover_no": 0, "games": 0}
    sub = team_log_df.copy()
    sub = sub[sub.get("team_key", "") == team_abbr]
    if "date" in sub.columns:
        sub = sub.sort_values("date")
    if last_n and last_n > 0:
        sub = sub.tail(last_n)

    ou_over = ou_under = 0
    if "ou_result_norm" in sub.columns:
        vals = sub["ou_result_norm"].dropna().astype(str).str.lower()
        ou_over = int((vals == "over").sum())
        ou_under = int((vals == "under").sum())

    cover_yes = cover_no = 0
    if "cover_result_norm" in sub.columns:
        cvals = sub["cover_result_norm"].dropna().astype(str).str.lower()
        cover_yes = int((cvals == "covered").sum())
        cover_no = int((cvals == "did not cover").sum())

    games = len(sub)
    return {"ou_over": ou_over, "ou_under": ou_under, "cover_yes": cover_yes, "cover_no": cover_no, "games": games}


def ou_adjustment_from_rates(team_log_df, home_abbr, away_abbr, weight_points: float = 4.0, last_n: int = 0):
    def rate(team):
        s = team_log_summary(team_log_df, team, last_n=last_n)
        tot = s["ou_over"] + s["ou_under"]
        if tot == 0:
            return 0.0
        return (s["ou_over"] / tot) - 0.5

    adj_home = rate(home_abbr) * weight_points
    adj_away = rate(away_abbr) * weight_points
    return (adj_home + adj_away) / 2.0


def spread_adjustment_from_rates(team_log_df, home_abbr, away_abbr, weight_points: float = 3.0, last_n: int = 0):
    def bias(team):
        s = team_log_summary(team_log_df, team, last_n=last_n)
        tot = s["cover_yes"] + s["cover_no"]
        if tot == 0:
            return 0.0
        return (s["cover_yes"] / tot) - 0.5

    home_bias = bias(home_abbr) * weight_points
    away_bias = bias(away_abbr) * weight_points
    return home_bias - away_bias


# ===== Player prop helpers =====
def find_player_across_sheets(player_name: str, qb_df, wr_df, rb_df, te_df):
    name = str(player_name).strip().lower()
    for df, pos in [(qb_df, "qb"), (rb_df, "rb"), (wr_df, "wr"), (te_df, "te")]:
        if "player" not in df.columns:
            continue
        mask = df["player"].astype(str).str.lower() == name
        if mask.any():
            return df[mask].copy(), pos
    return None, None


def get_games_played(row: pd.Series) -> float:
    if "games_played" in row.index and pd.notna(row["games_played"]):
        try:
            return float(row["games_played"])
        except Exception:
            pass
    return 1.0


def prop_stat_candidates(selected_prop: str):
    if selected_prop == "passing_yards":
        return ["yards_gained_by_passing", "passing_yards_gained"]
    elif selected_prop == "rushing_yards":
        return ["rushing_yards_gained"]
    elif selected_prop == "receiving_yards":
        return ["receiving_yards"]
    elif selected_prop == "receptions":
        return ["receptions"]
    elif selected_prop == "targets":
        return ["pass_targets"]
    elif selected_prop == "carries":
        return ["rushing_attempts"]
    return []


def get_stat_value_for_prop(row: pd.Series, selected_prop: str):
    for c in prop_stat_candidates(selected_prop):
        if c in row.index and pd.notna(row[c]):
            try:
                return float(row[c])
            except Exception:
                continue
    return None


def player_recent_avg(player_game_log_df: pd.DataFrame, player_name: str, selected_prop: str, last_n: int):
    if player_game_log_df is None or player_game_log_df.empty:
        return None
    name = str(player_name).strip().lower()
    sub = player_game_log_df[player_game_log_df["player"].astype(str).str.lower() == name].copy()
    if sub.empty:
        return None
    if "date" in sub.columns:
        sub = sub.sort_values("date")
    if last_n and last_n > 0:
        sub = sub.tail(last_n)
    if sub.empty:
        return None

    for c in prop_stat_candidates(selected_prop):
        if c in sub.columns:
            try:
                return float(pd.to_numeric(sub[c], errors="coerce").mean())
            except Exception:
                continue
    return None


def get_def_factor(team_log_df: pd.DataFrame, def_team_key: str, selected_prop: str):
    if team_log_df is None or team_log_df.empty or not def_team_key:
        return 1.0

    if selected_prop in ["passing_yards", "receiving_yards", "receptions", "targets"]:
        col = "passing_yards_gained_by_opposition"
    elif selected_prop in ["rushing_yards", "carries"]:
        col = "rushing_yards_by_opposition"
    else:
        return 1.0

    if col not in team_log_df.columns:
        return 1.0

    df = team_log_df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")

    league_allowed_pg = df[col].mean(skipna=True)
    if not np.isfinite(league_allowed_pg) or league_allowed_pg <= 0:
        return 1.0

    opp_rows = df[df["team_key"] == def_team_key]
    if opp_rows.empty:
        return 1.0

    opp_allowed_pg = opp_rows[col].mean(skipna=True)
    if not np.isfinite(opp_allowed_pg) or opp_allowed_pg <= 0:
        return 1.0

    factor = opp_allowed_pg / league_allowed_pg
    return float(np.clip(factor, 0.4, 1.6))


def prop_prediction_and_probs(
    player_name: str,
    selected_prop: str,
    line_val: float,
    selected_team_key: str,
    opponent_key: str,
    qb_df: pd.DataFrame,
    wr_df: pd.DataFrame,
    rb_df: pd.DataFrame,
    te_df: pd.DataFrame,
    player_game_log_df: pd.DataFrame,
    team_log_df: pd.DataFrame,
    last_n_games: int = PROP_TREND_LAST_N,
):
    player_df, pos = find_player_across_sheets(player_name, qb_df, wr_df, rb_df, te_df)
    if player_df is None or player_df.empty:
        return {"error": "Player not found in any of the QB/WR/RB/TE tables."}

    row = player_df.iloc[0]
    games_played = get_games_played(row)

    if selected_prop == "anytime_td":
        td_total = 0.0
        for c in ["rushing_touchdowns", "receiving_touchdowns", "passing_touchdowns"]:
            if c in row.index and pd.notna(row[c]):
                try:
                    td_total += float(row[c])
                except Exception:
                    continue

        if games_played <= 0 or td_total <= 0:
            return {"error": "No touchdown data found for this player."}

        td_rate = td_total / games_played
        prob_anytime = 1 - np.exp(-td_rate)
        prob_anytime = float(np.clip(prob_anytime, 0.0, 1.0))

        return {
            "prob_anytime": prob_anytime,
            "td_rate": td_rate,
            "td_total": td_total,
            "games_played": games_played,
        }

    season_total = get_stat_value_for_prop(row, selected_prop)
    if season_total is None:
        return {"error": "No matching stat column found for this prop in the season sheets."}

    season_pg = season_total / games_played if games_played > 0 else 0.0

    recent_pg = player_recent_avg(player_game_log_df, player_name, selected_prop, last_n_games)
    if recent_pg is None or season_pg <= 0:
        trend_factor = 1.0
    else:
        trend_factor = float(np.clip(recent_pg / season_pg, 0.4, 1.8))

    player_team_key = str(row.get("team_key", "")).strip()
    if selected_team_key:
        if player_team_key and player_team_key == selected_team_key:
            def_team_key = opponent_key
        else:
            def_team_key = selected_team_key
    else:
        def_team_key = opponent_key

    def_factor = get_def_factor(team_log_df, def_team_key, selected_prop)

    overall_adj_factor = trend_factor * def_factor
    predicted_pg = season_pg * overall_adj_factor

    stdev = max(3.0, predicted_pg * 0.35)
    z = (line_val - predicted_pg) / stdev
    prob_over = float(np.clip(1 - norm.cdf(z), 0.001, 0.999))
    prob_under = float(np.clip(norm.cdf(z), 0.001, 0.999))

    return {
        "predicted_pg": predicted_pg,
        "prob_over": prob_over,
        "prob_under": prob_under,
        "season_pg": season_pg,
        "recent_pg": recent_pg,
        "season_total": season_total,
        "games_played": games_played,
        "trend_factor": trend_factor,
        "def_factor": def_factor,
        "overall_adj_factor": overall_adj_factor,
        "player_team_key": player_team_key,
        "def_team_key": def_team_key,
    }


# Odds + market helpers
def american_to_decimal(odds: float) -> float:
    try:
        o = float(odds)
    except Exception:
        return np.nan
    return 1 + (o / 100.0) if o > 0 else 1 + (100.0 / abs(o))


def decimal_to_american(dec: float) -> float:
    if dec <= 1:
        return np.nan
    if dec >= 2:
        return round((dec - 1) * 100)
    return round(-100 / (dec - 1))


def prob_to_decimal(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return 1.0 / (1.0 - p + 1e-12)


def prob_to_american(p: float) -> float:
    dec = prob_to_decimal(p)
    return decimal_to_american(dec)


def prob_total_over_under(scores_df: pd.DataFrame, home: str, away: str, line_total: float):
    home_pts, away_pts = predict_scores(scores_df, home, away)
    pred_total = home_pts + away_pts
    stdev_total = max(6.0, pred_total * 0.18)
    z = (line_total - pred_total) / stdev_total
    p_over = float(np.clip(1 - norm.cdf(z), 0.001, 0.999))
    p_under = float(np.clip(norm.cdf(z), 0.001, 0.999))
    return pred_total, p_over, p_under


def prob_spread_cover(scores_df: pd.DataFrame, home: str, away: str, home_spread: float, side: str):
    home_pts, away_pts = predict_scores(scores_df, home, away)
    pred_margin = home_pts - away_pts
    stdev_margin = max(5.0, abs(pred_margin) * 0.9 + 6.0)
    line_margin = -home_spread
    z = (line_margin - pred_margin) / stdev_margin
    p_home_cover = float(np.clip(1 - norm.cdf(z), 0.001, 0.999))
    p_away_cover = 1.0 - p_home_cover
    return (pred_margin, p_home_cover) if side == "home" else (pred_margin, p_away_cover)


# =========================
# Prop type normalization for bulk upload
# =========================
def normalize_prop_type(raw: str):
    """
    Map various user-facing strings to internal prop keys.
    Examples that will now work:
    - 'Receiving Yards', 'REC_YDS' -> 'receiving_yards'
    - 'Rush Yds', 'Rushing Yards' -> 'rushing_yards'
    - 'Pass Yds', 'Passing Yards' -> 'passing_yards'
    - 'Receptions', 'Recs' -> 'receptions'
    - 'Carries', 'Rush Attempts', 'Rushing Attempts' -> 'carries'
    - 'Targets' -> 'targets'
    - 'Anytime TD', 'TD', 'Anytime Touchdown' -> 'anytime_td'
    """
    if pd.isna(raw):
        return None
    s = str(raw).strip().lower()
    # unify spaces/underscores
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    key = s.replace(" ", "_")

    # direct matches first
    allowed = {
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "receptions",
        "targets",
        "carries",
        "anytime_td",
    }
    if key in allowed:
        return key

    # synonyms
    synonyms = {
        "pass_yards": "passing_yards",
        "pass_yds": "passing_yards",
        "passing_yds": "passing_yards",
        "rush_yards": "rushing_yards",
        "rush_yds": "rushing_yards",
        "rushing_yds": "rushing_yards",
        "rec_yards": "receiving_yards",
        "rec_yds": "receiving_yards",
        "receiving_yds": "receiving_yards",
        "recs": "receptions",
        "receptions_": "receptions",
        "rush_attempts": "carries",
        "rushing_attempts": "carries",
        "rushes": "carries",
        "attempts": "carries",
        "targets_": "targets",
        "anytime_td_": "anytime_td",
        "anytime_touchdown": "anytime_td",
        "td": "anytime_td",
        "tds": "anytime_td",
    }

    if key in synonyms:
        return synonyms[key]

    # very simple fuzzy-ish handling
    if "receiv" in key and "yard" in key:
        return "receiving_yards"
    if "rush" in key and "yard" in key:
        return "rushing_yards"
    if ("pass" in key or "throw" in key) and "yard" in key:
        return "passing_yards"
    if "rec" in key and ("count" in key or "catch" in key or "reception" in key):
        return "receptions"
    if ("rush" in key or "carry" in key) and ("att" in key or "attempt" in key or "rushes" in key):
        return "carries"
    if "target" in key:
        return "targets"
    if "anytime" in key or ("td" in key and "yard" not in key):
        return "anytime_td"

    return None


# =========================
# UI – Main
# =========================
st.title("🏈 The Official un-official NFL Betting Model")

with st.expander("📘 How This Model Works", expanded=False):
    st.markdown("""
- **Team projections:** based on historical scoring, calibrated to league averages, plus optional log-based adjustments.
- **Player props:**
  - Season per-game → baseline.
  - **Last 2 games** → form/trend factor.
  - **Opponent defense** (yards allowed this year) → matchup factor.
  - Adjusted prediction = Season PG × Trend Factor × Defense Factor.
- **My Bets:** save singles and parlays as you go; see their probabilities together.
""")

with st.expander("📱 Add This App to Your Home Screen (Recommended)", expanded=False):
    st.markdown("""
**iPhone/iPad:** Share → Add to Home Screen  
**Android (Chrome):** ⋮ → Add to Home Screen
""")

scores_df = load_scores()
if scores_df.empty:
    st.error("Could not load NFL game data.")
    st.stop()

player_data = load_all_player_dfs()
qb_df = player_data["qb"]
wr_df = player_data["wr"]
rb_df = player_data["rb"]
te_df = player_data["te"]

player_game_log_df = load_player_game_log(PLAYER_GAME_LOG_URL)
team_log_df = load_team_game_log(TEAM_GAME_LOG_URL)

section_names = [
    "1) Game Selection + Prediction",
    "2) Top Edges This Week",
    "3) Player Props",
    "4) Parlay Builder (Players + Game Markets)",
    "5) Bulk Props Upload & Auto Parlay",
    "6) My Bets",
    "7) Team Game Log & Trends (NEW)"
]
selected_section = st.selectbox("Jump to section", section_names, index=0, help="Pick a section to open")


# -------------------------
# Section 1: Game Selection + Prediction
# -------------------------
with st.expander("1) Game Selection + Prediction", expanded=(selected_section == section_names[0])):
    st.subheader("Select Game")
    cols = st.columns([1, 1, 2])
    with cols[0]:
        week_vals = sorted(scores_df["week"].dropna().unique())
        selected_week = st.selectbox("Week", week_vals, key="sec1_week")
    with cols[1]:
        teams_in_week = sorted(
            set(scores_df.loc[scores_df["week"] == selected_week, "home_team"].dropna().unique())
            | set(scores_df.loc[scores_df["week"] == selected_week, "away_team"].dropna().unique())
        )
        selected_team = st.selectbox("Team", teams_in_week, key="sec1_team")

    game_row = scores_df[
        ((scores_df["home_team"] == selected_team) | (scores_df["away_team"] == selected_team))
        & (scores_df["week"] == selected_week)
    ]
    if game_row.empty:
        st.warning("No game found for that team/week.")
    else:
        g = game_row.iloc[0]
        home_team = g["home_team"]
        away_team = g["away_team"]
        opponent = away_team if home_team == selected_team else home_team

        with cols[2]:
            st.markdown(f"**Matchup:** {away_team} @ {home_team}")

        default_ou = float(g["over_under"]) if pd.notna(g.get("over_under", np.nan)) else 45.0
        default_home_spread = float(g["home_spread"]) if pd.notna(g.get("home_spread", np.nan)) else 0.0

        cL, cR = st.columns(2)
        with cL:
            over_under = st.number_input("Over/Under (Vegas or yours)", value=default_ou, step=0.5, key="sec1_ou")
        with cR:
            home_spread_val = st.number_input("Home-based Spread (home team perspective)", value=default_home_spread, step=0.5, key="sec1_spread")

        base_home_pts, base_away_pts = predict_scores(scores_df, home_team, away_team)
        base_total = base_home_pts + base_away_pts
        base_margin = base_home_pts - base_away_pts

        st.markdown("### Team Log Adjustments")
        use_adj = st.checkbox(
            "Blend in this season's O/U & Cover tendencies from the Team Game Log",
            value=True,
        )
        last_n_for_adj = st.number_input(
            "Use last N games for adjustments (0 = all)",
            min_value=0, value=0, step=1,
        )
        ou_weight = st.slider(
            "O/U tendency weight (± points to total at extremes)",
            0.0, 8.0, 4.0, 0.5,
        )
        spread_weight = st.slider(
            "Cover tendency weight (± points to margin at extremes)",
            0.0, 6.0, 3.0, 0.5,
        )

        adj_total = base_total
        adj_margin = base_margin
        if use_adj and team_log_df is not None and not team_log_df.empty:
            home_key = team_key(home_team)
            away_key = team_key(away_team)
            total_bump = ou_adjustment_from_rates(team_log_df, home_key, away_key, ou_weight, last_n_for_adj)
            margin_bump = spread_adjustment_from_rates(team_log_df, home_key, away_key, spread_weight, last_n_for_adj)
            adj_total = base_total + total_bump
            adj_margin = base_margin + margin_bump

        adj_home_pts = max(0.0, (adj_total + adj_margin) / 2.0)
        adj_away_pts = max(0.0, (adj_total - adj_margin) / 2.0)

        st.subheader("Predictions")
        mrow0 = st.columns(2)
        mrow0[0].metric("Base Predicted Total", f"{base_total:.1f}")
        mrow0[1].metric("Base Predicted Margin (Home - Away)", f"{base_margin:+.1f}")

        mrow1 = st.columns(2)
        mrow1[0].metric(f"{home_team} Predicted (Adjusted)", f"{adj_home_pts:.1f} pts")
        mrow1[1].metric(f"{away_team} Predicted (Adjusted)", f"{adj_away_pts:.1f} pts")

        adj_total_diff = adj_total - over_under
        adj_spread_diff = adj_margin - (-home_spread_val)

        mrow2 = st.columns(2)
        mrow2[0].metric("Adjusted Predicted Total", f"{adj_total:.1f}", f"{adj_total_diff:+.1f} vs O/U")
        mrow2[1].metric("Adjusted Predicted Margin", f"{adj_margin:+.1f}", f"{adj_spread_diff:+.1f} vs Home Spread")

        fig_total = px.bar(x=["Adjusted Total", "Vegas O/U"], y=[adj_total, over_under], title="Adjusted Total vs O/U")
        st.plotly_chart(fig_total, use_container_width=True)
        fig_margin = px.bar(x=["Adjusted Margin", "Home Spread Target"], y=[adj_margin, -home_spread_val], title="Adjusted Margin vs Home Spread")
        st.plotly_chart(fig_margin, use_container_width=True)


# -------------------------
# Section 2: Top Edges This Week
# -------------------------
with st.expander("2) Top Edges This Week", expanded=(selected_section == section_names[1])):
    week_list_all = sorted(scores_df["week"].dropna().unique())
    default_week = week_list_all[0] if week_list_all else None
    selected_week_for_edges = st.session_state.get("sec1_week", default_week)
    st.caption(f"Week shown: {selected_week_for_edges} — Legend: 🟩 strong | 🟨 lean | 🟥 pass")

    wk = scores_df[scores_df["week"] == selected_week_for_edges].copy()
    if "home_spread" not in wk.columns:
        wk["home_spread"] = wk.apply(compute_home_spread, axis=1)

    def strength_badge(edge_val):
        if pd.isna(edge_val):
            return "⬜"
        a = abs(edge_val)
        if a >= 4:
            return "🟩"
        elif a >= 2:
            return "🟨"
        else:
            return "🟥"

    rows = []
    for _, r in wk.iterrows():
        h, a = r.get("home_team"), r.get("away_team")
        if pd.isna(h) or pd.isna(a):
            continue
        h_pts, a_pts = predict_scores(scores_df, h, a)
        tot_pred = h_pts + a_pts
        mar_pred = h_pts - a_pts

        ou = float(r.get("over_under")) if pd.notna(r.get("over_under", np.nan)) else np.nan
        home_spread = float(r.get("home_spread")) if pd.notna(r.get("home_spread", np.nan)) else np.nan

        total_edge = np.nan if pd.isna(ou) else (tot_pred - ou)
        spread_edge = np.nan if pd.isna(home_spread) else (mar_pred - (-home_spread))

        if pd.isna(total_edge):
            total_pick = ""
        else:
            direction = "OVER" if total_edge > 0 else "UNDER"
            total_pick = f"{strength_badge(total_edge)} {direction}"

        if pd.isna(spread_edge) or pd.isna(home_spread):
            spread_pick = ""
        else:
            home_covers = mar_pred > -home_spread
            pick_text = f"{h} {home_spread:+.1f}" if home_covers else f"{a} {(-home_spread):+.1f}"
            spread_pick = f"{strength_badge(spread_edge)} {pick_text}"

        rows.append({
            "Matchup": f"{a} @ {h}",
            "Pred Total": round(tot_pred, 1),
            "O/U": ou if not pd.isna(ou) else "",
            "Total Edge (pts)": None if pd.isna(total_edge) else round(total_edge, 1),
            "Total Pick": total_pick,
            "Pred Margin": round(mar_pred, 1),
            "Home Spread": home_spread if not pd.isna(home_spread) else "",
            "Spread Edge (pts)": None if pd.isna(spread_edge) else round(spread_edge, 1),
            "Spread Pick": spread_pick,
        })

    edges_df = pd.DataFrame(rows)
    if not edges_df.empty:
        def best_edge(row):
            vals = [abs(v) for v in [row.get("Total Edge (pts)"), row.get("Spread Edge (pts)")] if pd.notna(v)]
            return max(vals) if vals else 0.0

        edges_df["Rank Score"] = edges_df.apply(best_edge, axis=1)
        edges_df = edges_df.sort_values("Rank Score", ascending=False).drop(columns=["Rank Score"])
        display_cols = ["Matchup", "Pred Total", "O/U", "Total Edge (pts)", "Total Pick", "Pred Margin", "Home Spread", "Spread Edge (pts)", "Spread Pick"]
        st.dataframe(edges_df[display_cols], use_container_width=True)
    else:
        st.info("No games found for this week.")


# -------------------------
# Section 3: Player Props
# -------------------------
with st.expander("3) Player Props", expanded=(selected_section == section_names[2])):
    if "sec1_team" not in st.session_state or "sec1_week" not in st.session_state:
        st.info("Pick a game in Section 1 first.")
    else:
        selected_team = st.session_state["sec1_team"]
        selected_week = st.session_state["sec1_week"]
        game_row = scores_df[
            ((scores_df["home_team"] == selected_team) | (scores_df["away_team"] == selected_team))
            & (scores_df["week"] == selected_week)
        ]
        if game_row.empty:
            st.warning("No game found for that team/week.")
        else:
            g = game_row.iloc[0]
            home_team = g["home_team"]
            away_team = g["away_team"]
            opponent = away_team if home_team == selected_team else home_team

            def players_for_team(df: pd.DataFrame, team_name_or_label: str):
                if "player" not in df.columns:
                    return []
                key = team_key(team_name_or_label)

                if "team_key" in df.columns:
                    mask = df["team_key"] == key
                else:
                    mask = pd.Series(False, index=df.index)

                if mask.any():
                    return list(df.loc[mask, "player"].dropna().unique())

                pfr_from_std = {
                    "NE": "NWE",
                    "GB": "GNB",
                    "KC": "KAN",
                    "TB": "TAM",
                    "NO": "NOR",
                    "SF": "SFO",
                    "LV": "LVR",
                    "LAR": "RAM",
                }
                df_team_upper = df.get("team", pd.Series("", index=df.index)).astype(str).str.upper()

                if key in pfr_from_std:
                    pfr_code = pfr_from_std[key]
                    mask_pfr = df_team_upper == pfr_code
                    if mask_pfr.any():
                        return list(df.loc[mask_pfr, "player"].dropna().unique())

                mask_std = df_team_upper == key
                if mask_std.any():
                    return list(df.loc[mask_std, "player"].dropna().unique())

                return []

            team_players = set(
                players_for_team(qb_df, selected_team)
                + players_for_team(rb_df, selected_team)
                + players_for_team(wr_df, selected_team)
                + players_for_team(te_df, selected_team)
            )
            opp_players = set(
                players_for_team(qb_df, opponent)
                + players_for_team(rb_df, opponent)
                + players_for_team(wr_df, opponent)
                + players_for_team(te_df, opponent)
            )
            both_players = sorted(team_players.union(opp_players))

            if not both_players:
                st.info(f"No players found for this matchup. Keys — You: **{team_key(selected_team)}**, Opp: **{team_key(opponent)}**.")

            c1, c2, c3 = st.columns([2, 1.2, 1.2])
            with c1:
                player_name = st.selectbox("Select Player", [""] + both_players, key="player_pick_props")
            with c2:
                prop_choices = ["passing_yards", "rushing_yards", "receiving_yards", "receptions", "targets", "carries", "anytime_td"]
                selected_prop = st.selectbox("Prop Type", prop_choices, index=2, key="prop_type_props")
            with c3:
                default_line = 50.0 if selected_prop != "anytime_td" else 0.0
                line_val = st.number_input("Sportsbook Line", value=float(default_line), key="prop_line") if selected_prop != "anytime_td" else 0.0

            if player_name:
                res = prop_prediction_and_probs(
                    player_name=player_name,
                    selected_prop=selected_prop,
                    line_val=line_val,
                    selected_team_key=team_key(selected_team),
                    opponent_key=team_key(opponent),
                    qb_df=qb_df,
                    wr_df=wr_df,
                    rb_df=rb_df,
                    te_df=te_df,
                    player_game_log_df=player_game_log_df,
                    team_log_df=team_log_df,
                    last_n_games=PROP_TREND_LAST_N,
                )

                if "error" in res:
                    st.warning(res["error"])
                elif selected_prop == "anytime_td":
                    st.subheader("Anytime TD Probability")
                    st.write(f"Estimated Anytime TD Probability: **{res['prob_anytime']*100:.1f}%**")
                    bar_df = pd.DataFrame({"Category": ["TDs/Game"], "Value": [res["td_rate"]]})
                    st.plotly_chart(
                        px.bar(bar_df, x="Category", y="Value", title=f"{player_name} – Anytime TD"),
                        use_container_width=True,
                    )
                else:
                    st.subheader(selected_prop.replace("_", " ").title())
                    st.write(f"**Season Total:** {res['season_total']:.1f}")
                    st.write(f"**Games Played:** {res['games_played']:.0f}")
                    st.write(f"**Per Game (season):** {res['season_pg']:.2f}")
                    if res["recent_pg"] is not None:
                        st.write(f"**Last {PROP_TREND_LAST_N} games (avg):** {res['recent_pg']:.2f}")
                    st.write(f"**Trend factor (last {PROP_TREND_LAST_N} vs season):** {res['trend_factor']:.3f}")

                    # Nice label: use opponent's full name if possible
                    def_team_code = res.get("def_team_key", "")
                    def_team_label = CODE_TO_FULLNAME.get(def_team_code, opponent)
                    st.write(f"**Defense factor (vs {def_team_label}):** {res['def_factor']:.3f}")
                    st.write(f"**Overall adjustment factor:** {res['overall_adj_factor']:.3f}")
                    st.write(f"**Adjusted prediction (this game):** {res['predicted_pg']:.2f}")
                    st.write(f"**Line:** {line_val:.1f}")
                    st.write(f"**Probability of OVER:** {res['prob_over']*100:.1f}%")
                    st.write(f"**Probability of UNDER:** {res['prob_under']*100:.1f}%")

                    fig = px.bar(
                        x=["Season PG", f"Last {PROP_TREND_LAST_N} PG", "Adjusted Prediction", "Line"],
                        y=[
                            res["season_pg"],
                            res["recent_pg"] if res["recent_pg"] is not None else 0.0,
                            res["predicted_pg"],
                            line_val,
                        ],
                        title=f"{player_name} – {selected_prop.replace('_', ' ').title()}",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # ----- Add to My Bets -----
                    add_col1, add_col2 = st.columns([2, 1])
                    with add_col1:
                        st.markdown("### Save to My Bets")
                    with add_col2:
                        if st.button("➕ Add this prop to My Bets", key="add_prop_to_bets"):
                            side = "over" if res["prob_over"] >= res["prob_under"] else "under"
                            model_prob = float(res["prob_over"] if side == "over" else res["prob_under"])
                            try:
                                game_week = int(g["week"]) if "week" in g.index and pd.notna(g["week"]) else None
                            except Exception:
                                game_week = None

                            bet_entry = {
                                "type": "player_prop",
                                "game": f"{away_team} @ {home_team}",
                                "week": game_week,
                                "player": player_name,
                                "team": selected_team,
                                "opponent": opponent,
                                "prop": selected_prop,
                                "side": side,
                                "line": float(line_val),
                                "model_prob": model_prob,
                                "season_pg": res["season_pg"],
                                "recent_pg": res["recent_pg"],
                                "predicted_pg": res["predicted_pg"],
                            }
                            st.session_state["my_bets"].append(bet_entry)
                            st.success("Added to My Bets ✅")

            else:
                st.info("Select a player to evaluate props.")


# -------------------------
# Section 4: Parlay Builder
# -------------------------
with st.expander("4) Parlay Builder (Players + Game Markets)", expanded=(selected_section == section_names[3])):
    if "parlay_legs" not in st.session_state:
        st.session_state["parlay_legs"] = []

    def unique_players(*dfs):
        names = []
        for df in dfs:
            if "player" in df.columns:
                names.extend(list(df["player"].dropna().astype(str).unique()))
        return sorted(pd.unique(names))

    all_players = unique_players(qb_df, wr_df, rb_df, te_df)
    full_team_names = sorted(list(CODE_TO_FULLNAME.values()))

    st.markdown("**Add Player Prop Leg**")
    a1, a2, a3, a4 = st.columns([2.2, 1.4, 1.4, 2])
    with a1:
        pb_player = st.selectbox("Player", [""] + all_players, key="pb_any_player")
    with a2:
        pb_prop = st.selectbox("Prop", ["passing_yards", "rushing_yards", "receiving_yards", "receptions", "targets", "carries", "anytime_td"], key="pb_any_prop")
    with a3:
        if pb_prop == "anytime_td":
            pb_line = 0.0
            st.text_input("Line", "—", disabled=True, key="pb_any_line_disabled")
        else:
            pb_line = st.number_input("Line", value=50.0, step=0.5, key="pb_any_line")
    with a4:
        pb_opp_full = st.selectbox("Opponent (Full Name)", full_team_names, key="pb_any_opp_full")
        pb_opp_key = FULLNAME_TO_CODE.get(pb_opp_full, "")

    if st.button("➕ Add Player Leg", use_container_width=True, key="pb_any_add"):
        if not pb_player:
            st.warning("Pick a player first.")
        elif not pb_opp_key:
            st.warning("Pick an opponent team.")
        else:
            res = prop_prediction_and_probs(
                player_name=pb_player,
                selected_prop=pb_prop,
                line_val=pb_line,
                selected_team_key="",
                opponent_key=pb_opp_key,
                qb_df=qb_df,
                wr_df=wr_df,
                rb_df=rb_df,
                te_df=te_df,
                player_game_log_df=player_game_log_df,
                team_log_df=team_log_df,
                last_n_games=PROP_TREND_LAST_N,
            )
            if "error" in res:
                st.warning(res["error"])
            else:
                if pb_prop == "anytime_td":
                    prob = float(res["prob_anytime"])
                    side = "yes"
                    label = f"{pb_player} Anytime TD vs {pb_opp_full}"
                else:
                    side = "over" if res["prob_over"] >= res["prob_under"] else "under"
                    prob = float(res["prob_over"] if side == "over" else res["prob_under"])
                    label = f"{pb_player} {pb_prop.replace('_',' ').title()} {side.title()} {pb_line} vs {pb_opp_full}"

                st.session_state["parlay_legs"].append({
                    "kind": "player",
                    "label": label,
                    "prob": prob,
                    "player": pb_player,
                    "prop": pb_prop,
                    "line": float(pb_line),
                    "side": side,
                    "opponent": pb_opp_full,
                })
                st.success("Added leg to parlay ✅")

    st.markdown("---")

    st.markdown("**Add Game Market Leg**")
    g1, g2, g3, g4 = st.columns([1.0, 2.4, 1.6, 2])
    with g1:
        week_for_market = st.selectbox("Week", sorted(scores_df["week"].dropna().unique()), key="gm_week")
    with g2:
        wk_df = scores_df[scores_df["week"] == week_for_market].copy()
        if "home_spread" not in wk_df.columns:
            wk_df["home_spread"] = wk_df.apply(compute_home_spread, axis=1)

        matchups, meta, home_spreads_for_match, ous_for_match = [], [], [], []
        for _, row in wk_df.iterrows():
            h = row.get("home_team")
            a = row.get("away_team")
            if pd.isna(h) or pd.isna(a):
                continue
            matchups.append(f"{a} @ {h}")
            meta.append((h, a))
            home_spreads_for_match.append(float(row.get("home_spread")) if pd.notna(row.get("home_spread", np.nan)) else 0.0)
            ous_for_match.append(float(row.get("over_under")) if pd.notna(row.get("over_under", np.nan)) else 45.0)

        gm_match = st.selectbox("Matchup", matchups, key="gm_matchup")
        idx = matchups.index(gm_match) if gm_match in matchups else -1
        gm_home = meta[idx][0] if idx >= 0 else None
        gm_away = meta[idx][1] if idx >= 0 else None
        default_home_sp = home_spreads_for_match[idx] if idx >= 0 else 0.0
        default_ou = ous_for_match[idx] if idx >= 0 else 45.0
    with g3:
        gm_market = st.selectbox("Market", ["Total", "Spread"], key="gm_market")
    with g4:
        if gm_market == "Total":
            gm_total = st.number_input("O/U Line", value=float(default_ou), step=0.5, key="gm_total_line")
        else:
            gm_spread = st.number_input("Home-based Spread", value=float(default_home_sp), step=0.5, key="gm_spread_line")

    # Show model prob for this leg
    if gm_home and gm_away:
        if gm_market == "Total":
            _, p_over, p_under = prob_total_over_under(scores_df, gm_home, gm_away, gm_total)
            side = "over" if p_over >= p_under else "under"
            leg_prob = float(p_over if side == "over" else p_under)
            st.metric("Model side", f"{side.title()} ({leg_prob*100:.1f}%)")
        else:
            _, p_cov_home = prob_spread_cover(scores_df, gm_home, gm_away, gm_spread, "home")
            _, p_cov_away = prob_spread_cover(scores_df, gm_home, gm_away, gm_spread, "away")
            side = "home" if p_cov_home >= p_cov_away else "away"
            leg_prob = float(p_cov_home if side == "home" else p_cov_away)
            st.metric("Model side", f"{'Home' if side=='home' else 'Away'} ({leg_prob*100:.1f}%)")

        if st.button("➕ Add Game Leg", use_container_width=True, key="gm_add"):
            if gm_market == "Total":
                label = f"{gm_away} @ {gm_home} Total {side.title()} {gm_total}"
                st.session_state["parlay_legs"].append({
                    "kind": "game_total",
                    "label": label,
                    "prob": leg_prob,
                    "home_team": gm_home,
                    "away_team": gm_away,
                    "side": side,
                    "line": float(gm_total),
                })
            else:
                if side == "home":
                    side_text = f"{gm_home} {gm_spread:+.1f}"
                else:
                    side_text = f"{gm_away} {(-gm_spread):+.1f}"
                label = f"{gm_away} @ {gm_home} Spread {side_text}"
                st.session_state["parlay_legs"].append({
                    "kind": "game_spread",
                    "label": label,
                    "prob": leg_prob,
                    "home_team": gm_home,
                    "away_team": gm_away,
                    "side": side,
                    "line": float(gm_spread),
                })
            st.success("Added leg to parlay ✅")

    # Show current legs & overall parlay prob
    if st.session_state["parlay_legs"]:
        st.subheader("Current Parlay Legs")
        for i, leg in enumerate(st.session_state["parlay_legs"]):
            c1, c2 = st.columns([8, 1])
            c1.markdown(f"• **{leg.get('label','Leg')}** — Model Pr: **{leg.get('prob',0.0)*100:.1f}%**")
            if c2.button("🗑 Remove", key=f"rm_parlay_leg_{i}"):
                st.session_state["parlay_legs"].pop(i)
                st.rerun()

        probs = [float(leg.get("prob", 0.0)) for leg in st.session_state["parlay_legs"]]
        parlay_hit_prob = float(np.prod(probs)) if probs else 0.0
        model_dec_odds = prob_to_decimal(parlay_hit_prob)
        model_am_odds = prob_to_american(parlay_hit_prob)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Parlay Hit Probability", f"{parlay_hit_prob*100:.1f}%")
        with c2:
            st.metric("Model Fair Odds", f"{int(model_am_odds):+d}")

        if st.button("💾 Save this parlay to My Bets", key="save_parlay_to_bets"):
            parlay_bet = {
                "type": "parlay_custom",
                "label": f"Custom Parlay ({len(st.session_state['parlay_legs'])} legs)",
                "legs": list(st.session_state["parlay_legs"]),
                "model_prob": parlay_hit_prob,
            }
            st.session_state["my_bets"].append(parlay_bet)
            st.success("Saved parlay to My Bets ✅")

    else:
        st.info("Add legs above to build a parlay.")


# -------------------------
# Section 5: Bulk Props Upload & Auto Parlay
# -------------------------
with st.expander("5) Bulk Props Upload & Auto Parlay", expanded=(selected_section == section_names[4])):
    st.markdown("""
Upload a CSV with columns:

- **Player**  
- **Team** (their team; optional but nice)  
- **Opponent** (team they're facing)  
- **PropType** (one of: `passing_yards`, `rushing_yards`, `receiving_yards`, `receptions`, `targets`, `carries`, `anytime_td`)  
- **Line** (sportsbook line)

The model will:
1. Compute probabilities for **over** and **under**.
2. Pick the better side automatically.
3. Score each prop.
4. Build the highest-probability parlay using your chosen leg count.
""")

    uploaded = st.file_uploader("Upload bulk props CSV", type=["csv"], key="bulk_props_upload")
    if uploaded is not None:
        bulk_df = pd.read_csv(uploaded)
        required_cols = ["Player", "Opponent", "PropType", "Line"]
        missing = [c for c in required_cols if c not in bulk_df.columns]
        if missing:
            st.error(f"Missing columns in upload: {', '.join(missing)}")
        else:
            legs_candidates = []
            error_rows = []
            for idx, row in bulk_df.iterrows():
                player = str(row["Player"]).strip()
                opponent = str(row["Opponent"]).strip()
                raw_prop_type = row["PropType"]
                prop_type = normalize_prop_type(raw_prop_type)

                if not player or not opponent or prop_type is None:
                    error_rows.append({
                        "Row": idx + 1,
                        "Player": player,
                        "Opponent": opponent,
                        "PropType_raw": raw_prop_type,
                        "Error": "Invalid/missing player, opponent, or PropType"
                    })
                    continue

                try:
                    line_val = float(row["Line"])
                except Exception:
                    error_rows.append({
                        "Row": idx + 1,
                        "Player": player,
                        "Opponent": opponent,
                        "PropType_raw": raw_prop_type,
                        "Error": "Line could not be parsed as number"
                    })
                    continue

                opp_key = team_key(opponent)
                res = prop_prediction_and_probs(
                    player_name=player,
                    selected_prop=prop_type,
                    line_val=line_val,
                    selected_team_key="",
                    opponent_key=opp_key,
                    qb_df=qb_df,
                    wr_df=wr_df,
                    rb_df=rb_df,
                    te_df=te_df,
                    player_game_log_df=player_game_log_df,
                    team_log_df=team_log_df,
                    last_n_games=PROP_TREND_LAST_N,
                )
                if "error" in res:
                    error_rows.append({
                        "Row": idx + 1,
                        "Player": player,
                        "Opponent": opponent,
                        "PropType_raw": raw_prop_type,
                        "Error": res["error"],
                    })
                    continue

                if prop_type == "anytime_td":
                    side = "yes"
                    prob = float(res["prob_anytime"])
                    label = f"{player} Anytime TD vs {opponent}"
                else:
                    side = "over" if res["prob_over"] >= res["prob_under"] else "under"
                    prob = float(res["prob_over"] if side == "over" else res["prob_under"])
                    label = f"{player} {prop_type.replace('_',' ').title()} {side.title()} {line_val} vs {opponent}"

                legs_candidates.append({
                    "label": label,
                    "prob": prob,
                    "player": player,
                    "prop": prop_type,
                    "side": side,
                    "line": float(line_val),
                    "opponent": opponent,
                })

            if error_rows:
                st.subheader("Rows That Could Not Be Evaluated")
                st.dataframe(pd.DataFrame(error_rows), use_container_width=True)

            if not legs_candidates:
                st.warning("No valid props could be evaluated from this file.")
            else:
                cand_df = pd.DataFrame([{"Label": l["label"], "Model Prob %": l["prob"] * 100} for l in legs_candidates])
                st.subheader("All Evaluated Props")
                st.dataframe(cand_df, use_container_width=True)

                max_legs = min(10, len(legs_candidates))
                leg_count = st.slider("Number of legs in auto parlay", 2, max_legs, min(5, max_legs))
                legs_sorted = sorted(legs_candidates, key=lambda x: x["prob"], reverse=True)
                chosen_legs = legs_sorted[:leg_count]

                auto_probs = [l["prob"] for l in chosen_legs]
                auto_parlay_prob = float(np.prod(auto_probs))
                auto_dec_odds = prob_to_decimal(auto_parlay_prob)
                auto_am_odds = prob_to_american(auto_parlay_prob)

                st.subheader("Recommended Auto Parlay")
                st.metric("Parlay Hit Probability", f"{auto_parlay_prob*100:.1f}%")
                st.metric("Model Fair Odds", f"{int(auto_am_odds):+d}")

                st.markdown("**Legs**")
                auto_leg_df = pd.DataFrame(
                    [{"Leg": l["label"], "Model Prob %": l["prob"] * 100} for l in chosen_legs]
                )
                st.table(auto_leg_df)

                if st.button("💾 Save this auto parlay to My Bets", key="save_auto_parlay_to_bets"):
                    parlay_bet = {
                        "type": "parlay_auto",
                        "label": f"Auto Parlay ({len(chosen_legs)} legs)",
                        "legs": chosen_legs,
                        "model_prob": auto_parlay_prob,
                    }
                    st.session_state["my_bets"].append(parlay_bet)
                    st.success("Auto parlay saved to My Bets ✅")


# -------------------------
# Section 6: My Bets
# -------------------------
with st.expander("6) My Bets", expanded=(selected_section == section_names[5])):
    bets = st.session_state.get("my_bets", [])

    if not bets:
        st.info("You don't have any saved bets yet. Add them from **Player Props** or **Parlay Builder** or **Bulk Upload**.")
    else:
        singles = [b for b in bets if b.get("type") == "player_prop"]
        parlays = [b for b in bets if str(b.get("type", "")).startswith("parlay")]

        st.subheader("Singles (Player Props)")
        if singles:
            singles_df = pd.DataFrame(
                [
                    {
                        "Game": b.get("game", ""),
                        "Week": b.get("week", ""),
                        "Player": b.get("player", ""),
                        "Team": b.get("team", ""),
                        "Opponent": b.get("opponent", ""),
                        "Prop": b.get("prop", ""),
                        "Side": b.get("side", ""),
                        "Line": b.get("line", ""),
                        "Model Prob %": b.get("model_prob", 0.0) * 100,
                        "Adj Pred": b.get("predicted_pg", ""),
                    }
                    for b in singles
                ]
            )
            st.dataframe(singles_df, use_container_width=True)
        else:
            st.write("No singles saved yet.")

        st.subheader("Parlays")
        if parlays:
            for i, parlay in enumerate(parlays):
                label = parlay.get("label", f"Parlay {i+1}")
                p = parlay.get("model_prob", 0.0)
                st.markdown(f"### {label} — Model Prob: **{p*100:.1f}%**")
                legs = parlay.get("legs", [])
                if legs:
                    leg_rows = []
                    for j, leg in enumerate(legs, start=1):
                        leg_rows.append({
                            "Leg #": j,
                            "Description": leg.get("label", ""),
                            "Model Prob %": leg.get("prob", 0.0) * 100,
                        })
                    legs_df = pd.DataFrame(leg_rows)
                    st.table(legs_df)
                else:
                    st.write("_No legs recorded for this parlay (check how it was saved)._")
                st.markdown("---")
        else:
            st.write("No parlays saved yet.")

        st.markdown("### Export / Manage")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clear all My Bets", key="clear_my_bets"):
                st.session_state["my_bets"] = []
                st.success("Cleared all bets.")
                st.rerun()
        with col2:
            # Simple CSV export of all bets (flattened)
            flat_rows = []
            for b in bets:
                if b.get("type") == "player_prop":
                    flat_rows.append({
                        "Type": "Single",
                        "Label": f"{b.get('player','')} {b.get('prop','')} {b.get('side','')} {b.get('line','')}",
                        "Game": b.get("game",""),
                        "Week": b.get("week",""),
                        "Model Prob": b.get("model_prob",0.0),
                    })
                else:
                    label = b.get("label", "Parlay")
                    flat_rows.append({
                        "Type": "Parlay",
                        "Label": label,
                        "Game": "",
                        "Week": "",
                        "Model Prob": b.get("model_prob",0.0),
                    })
            if flat_rows:
                export_df = pd.DataFrame(flat_rows)
                csv_bytes = export_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download My Bets as CSV",
                    data=csv_bytes,
                    file_name="my_bets.csv",
                    mime="text/csv",
                )
            else:
                st.write("Nothing to export yet.")


# -------------------------
# Section 7: Team Game Log & Trends
# -------------------------
with st.expander("7) Team Game Log & Trends (NEW)", expanded=(selected_section == section_names[6])):
    if team_log_df is None or team_log_df.empty:
        st.info("No team game log data found.")
    else:
        teams = sorted([t for t in team_log_df["team"].dropna().unique()] if "team" in team_log_df.columns else [])
        if not teams:
            st.info("Team column not present in sheet. Check your sharing/export.")
        else:
            tcol1, tcol2, tcol3 = st.columns([1.2, 1, 1])
            with tcol1:
                pick_team = st.selectbox("Team (abbr.)", teams, key="tgl_team")
            with tcol2:
                day_filter = st.selectbox("Day filter", ["All", "Sunday", "Monday", "Thursday"], index=0, key="tgl_day")
            with tcol3:
                last_n = st.number_input("Last N games (0 = all)", value=0, min_value=0, step=1, key="tgl_lastn")

            sub = team_log_df.copy()
            sub = sub[sub["team"] == pick_team]

            if day_filter != "All" and "day" in sub.columns:
                sub = sub[sub["day"].astype(str).str.lower() == day_filter.lower()]

            if "date" in sub.columns:
                sub = sub.sort_values("date")
            if last_n and last_n > 0:
                sub = sub.tail(last_n)

            ou_over = ou_under = ou_total = 0
            if "ou_result_norm" in sub.columns:
                vals = sub["ou_result_norm"].dropna().astype(str).str.lower()
                ou_over = (vals == "over").sum()
                ou_under = (vals == "under").sum()
                ou_total = (vals.isin(["over", "under"])).sum()
            cover_yes = cover_no = cover_total = 0
            if "cover_result_norm" in sub.columns:
                cvals = sub["cover_result_norm"].dropna().astype(str).str.lower()
                cover_yes = (cvals == "covered").sum()
                cover_no = (cvals == "did not cover").sum()
                cover_total = (cvals.isin(["covered", "did not cover"])).sum()

            m1, m2, m3 = st.columns(3)
            with m1:
                pct_over = (ou_over / ou_total * 100.0) if ou_total else 0.0
                st.metric("O/U Over Hit %", f"{pct_over:.1f}%", f"{ou_over}/{ou_total}")
            with m2:
                pct_under = (ou_under / ou_total * 100.0) if ou_total else 0.0
                st.metric("O/U Under Hit %", f"{pct_under:.1f}%", f"{ou_under}/{ou_total}")
            with m3:
                pct_cover = (cover_yes / cover_total * 100.0) if cover_total else 0.0
                st.metric("Spread Cover %", f"{pct_cover:.1f}%", f"{cover_yes}/{cover_total} covered")

            if "day" in team_log_df.columns and "ou_result_norm" in team_log_df.columns:
                day_data = team_log_df[team_log_df["team"] == pick_team].copy()
                day_data = day_data[day_data["ou_result_norm"].isin(["over", "under"])]
                if not day_data.empty:
                    grp = day_data.groupby(["day", "ou_result_norm"]).size().reset_index(name="count")
                    fig = px.bar(
                        grp,
                        x="day",
                        y="count",
                        color="ou_result_norm",
                        title=f"{pick_team} — O/U Results by Day",
                        barmode="group",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            show_cols = []
            label_map = []
            if "date" in sub.columns:
                show_cols.append("date")
                label_map.append("Date")
            if "day" in sub.columns:
                show_cols.append("day")
                label_map.append("Day")
            if "opponent" in sub.columns:
                show_cols.append("opponent")
                label_map.append("Opponent")
            if "is_away" in sub.columns:
                show_cols.append("is_away")
                label_map.append("Away?")
            if "ou_line" in sub.columns:
                show_cols.append("ou_line")
                label_map.append("O/U Line")
            if "ou_result" in sub.columns:
                show_cols.append("ou_result")
                label_map.append("O/U Result")
            if "spread" in sub.columns:
                show_cols.append("spread")
                label_map.append("Spread")
            if "cover_result" in sub.columns:
                show_cols.append("cover_result")
                label_map.append("Cover Result")
            if "result" in sub.columns:
                show_cols.append("result")
                label_map.append("Result")

            if not show_cols:
                st.info("Table columns not found in sheet. Check headers.")
            else:
                st.subheader("Recent Games")
                pretty = sub[show_cols].copy()
                if "is_away" in pretty.columns:
                    pretty["is_away"] = pretty["is_away"].map({True: "@", False: "home"}).fillna("")
                if "date" in pretty.columns:
                    pretty["date"] = pd.to_datetime(pretty["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                pretty.columns = label_map
                st.dataframe(pretty, use_container_width=True)

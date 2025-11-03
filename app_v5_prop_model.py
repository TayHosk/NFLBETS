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

def find_player_in(df: pd.DataFrame, player_name: str):
    if "player" not in df.columns:
        return None
    mask = df["player"].astype(str).str.lower() == str(player_name).lower()
    return df[mask].copy() if mask.any() else None

# =========================
# UI – Single Page
# =========================
st.title("🏈 NFL Game + Player Prop Dashboard (One-Page)")

scores_df = load_scores()
player_data = load_all_player_dfs()
p_rec, p_rush, p_pass = player_data["player_receiving"], player_data["player_rushing"], player_data["player_passing"]
d_rb, d_qb, d_wr, d_te = player_data["def_rb"], player_data["def_qb"], player_data["def_wr"], player_data["def_te"]

# (Your Week, Game Prediction, and Top Edges sections remain unchanged)

# -------------------------
# 4) Player Props (players from both teams)
# -------------------------
with st.container():
    st.header("4) Player Props (Both Teams)")

    def players_for_team(df, team_name):
        if "team" not in df.columns or "player" not in df.columns:
            return []
        mask = df["team"].astype(str).str.lower() == str(team_name).lower()
        return list(df.loc[mask, "player"].dropna().unique())

    team_players = set(players_for_team(p_rec, selected_team) + players_for_team(p_rush, selected_team) + players_for_team(p_pass, selected_team))
    opp_players = set(players_for_team(p_rec, opponent) + players_for_team(p_rush, opponent) + players_for_team(p_pass, opponent))
    both_players = sorted(team_players.union(opp_players))

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
        player_df_source = p_rec if selected_prop in ["receiving_yards", "receptions", "targets"] else (
            p_rush if selected_prop in ["rushing_yards", "carries"] else p_pass
        )
        this_player_df = find_player_in(player_df_source, player_name)

        # --- FIXED ANYTIME TD SECTION ---
        if (this_player_df is None or this_player_df.empty) and selected_prop == "anytime_td":
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
                td_rate = total_tds / total_games

                player_team = None
                for df_ in [p_rec, p_rush, p_pass]:
                    row_ = find_player_in(df_, player_name)
                    if row_ is not None and not row_.empty and "team" in row_.columns:
                        player_team = str(row_.iloc[0]["team"])
                        break
                opp_team_for_player = opponent if str(player_team).lower() == str(selected_team).lower() else selected_team

                def_dfs = [d_rb.copy(), d_wr.copy(), d_te.copy()]
                for d in def_dfs:
                    if "games_played" not in d.columns:
                        d["games_played"] = 1
                    td_allowed_cols = [c for c in d.columns if "td" in c and "allowed" in c]
                    if td_allowed_cols:
                        d["tds_pg"] = d[td_allowed_cols].sum(axis=1) / d["games_played"].replace(0, np.nan)
                    else:
                        d["tds_pg"] = np.nan

                league_td_pg = np.nanmean([d["tds_pg"].mean() for d in def_dfs])
                opp_td_pg = np.nan
                for d in def_dfs:
                    mask = d["team"].astype(str).str.lower() == str(opp_team_for_player).lower()
                    if mask.any():
                        opp_td_pg = np.nanmean(d.loc[mask, "tds_pg"])
                        break
                if np.isnan(opp_td_pg):
                    opp_td_pg = league_td_pg

                adj_factor = (opp_td_pg / league_td_pg) if league_td_pg and league_td_pg > 0 else 1.0
                adj_td_rate = min(td_rate * adj_factor, 1.0)

                prob_anytime = 1 - np.exp(-adj_td_rate * 1.1)
                prob_anytime = float(np.clip(prob_anytime, 0.0, 1.0))

                st.subheader("🏈 Anytime TD Probability")
                st.write(f"**Estimated Probability:** {prob_anytime*100:.1f}% chance to score a TD")

                bar_df = pd.DataFrame({
                    "Category": ["Season TD Rate", "Adj. vs Opponent"],
                    "TDs/Game": [td_rate, adj_td_rate]
                })
                st.plotly_chart(
                    px.bar(
                        bar_df,
                        x="Category",
                        y="TDs/Game",
                        title=f"{player_name} – Anytime TD Probability vs {opp_team_for_player}",
                        color="Category",
                        color_discrete_sequence=["#a2d5f2", "#07689f"]
                    ),
                    use_container_width=True
                )
        # --- END FIXED ANYTIME TD SECTION ---

        elif this_player_df is None or this_player_df.empty:
            st.warning("Player not found in data.")
        else:
            st.write("Normal prop logic unchanged...")

    else:
        st.info("Select a player to evaluate props.")

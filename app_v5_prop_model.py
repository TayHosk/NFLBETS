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

# ===== Prop helpers (v7.7 logic) =====
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
    # Build edges per game
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

    # Build player list from both teams across receiving/rushing/passing sheets
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
        if selected_prop == "anytime_td":            # For anytime TD we may need both tables
            rec_row = find_player_in(p_rec, player_name)
            rush_row = find_player_in(p_rush, player_name)
            # compute TD rate from both
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
                st.warning("No games data found for this player.")
            else:
                # Defense context (RB/WR/TE)
                def_dfs = [d_rb.copy(), d_wr.copy(), d_te.copy()]
                for d in def_dfs:
                    if "games_played" not in d.columns:
                        d["games_played"] = 1
                    td_cols = [c for c in d.columns if "td" in c and "allowed" in c]
                    d["tds_pg"] = d[td_cols].sum(axis=1) / d["games_played"].replace(0, np.nan)

                league_td_pg = np.nanmean([d["tds_pg"].mean() for d in def_dfs])
                # Opponent is whichever team the player is facing
                # Determine player's team to infer opponent
                # (If player's team matches selected_team, opponent is opponent; else selected_team)
                player_team = None
                for df_ in [p_rec, p_rush, p_pass]:
                    row_ = find_player_in(df_, player_name)
                    if row_ is not None and not row_.empty and "team" in row_.columns:
                        player_team = str(row_.iloc[0]["team"])
                        break
                opp_team_for_player = opponent if str(player_team).lower() == str(selected_team).lower() else selected_team

                opp_td_list = []
                for d in def_dfs:
                    mask = d["team"].astype(str).str.lower() == str(opp_team_for_player).lower()
                    opp_td_list.append(d.loc[mask, "tds_pg"].mean())
                opp_td_pg = np.nanmean(opp_td_list)
                if np.isnan(opp_td_pg):
                    opp_td_pg = league_td_pg

                adj_factor = (opp_td_pg / league_td_pg) if league_td_pg and league_td_pg > 0 else 1.0
                adj_td_rate = (total_tds / total_games) * adj_factor
                prob_anytime = min(adj_td_rate, 1.0)
                st.subheader("Anytime TD Probability")
                st.write(f"Estimated Anytime TD Probability: **{prob_anytime*100:.1f}%**")
                bar_df = pd.DataFrame(
                    {"Category": ["Player TD Rate", "Adj. vs Opponent"], "TDs/Game": [(total_tds/total_games), adj_td_rate]}
                )
                st.plotly_chart(px.bar(bar_df, x="Category", y="TDs/Game", title=f"{player_name} – Anytime TD vs {opp_team_for_player}"), use_container_width=True)

        elif this_player_df is None or this_player_df.empty:
            st.warning("Player not found in the selected stat table.")
        else:
            # Normal props flow
            player_pos = this_player_df.iloc[0].get("position", fallback_pos)

            stat_col = detect_stat_col(this_player_df, selected_prop)
            if not stat_col and selected_prop != "anytime_td":
                st.warning("No matching stat column found for this prop.")
            else:
                # Season per-game
                season_val = float(this_player_df.iloc[0][stat_col]) if stat_col and pd.notna(this_player_df.iloc[0][stat_col]) else 0.0
                games_played = float(this_player_df.iloc[0].get("games_played", 1)) or 1.0
                player_pg = season_val / games_played if games_played > 0 else 0.0

                # Defense adjust
                def_df = pick_def_df(selected_prop, player_pos, d_qb, d_rb, d_wr, d_te)
                def_col = detect_def_col(def_df, selected_prop) if def_df is not None else None

                opp_allowed_pg = None
                league_allowed_pg = None
                # Determine player's team for opponent logic
                player_team = str(this_player_df.iloc[0].get("team", "")).strip()
                opp_team_for_player = opponent if player_team.lower() == str(selected_team).lower() else selected_team

                if def_df is not None and def_col is not None:
                    if "games_played" in def_df.columns:
                        league_allowed_pg = (def_df[def_col] / def_df["games_played"].replace(0, np.nan)).mean()
                    else:
                        league_allowed_pg = def_df[def_col].mean()

                    opp_row = def_df[def_df["team"].astype(str).str.lower() == opp_team_for_player.lower()]
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
                z = (line_val - predicted_pg) / stdev if selected_prop != "anytime_td" else 0.0
                prob_over = (1 - norm.cdf(z)) if selected_prop != "anytime_td" else np.nan
                prob_under = (norm.cdf(z)) if selected_prop != "anytime_td" else np.nan
                if selected_prop != "anytime_td":
                    prob_over = float(np.clip(prob_over, 0.001, 0.999))
                    prob_under = float(np.clip(prob_under, 0.001, 0.999))

                st.subheader(selected_prop.replace("_", " ").title())
                st.write(f"**Player (season total):** {season_val:.2f} over {games_played:.0f} games → **{player_pg:.2f} per game**")
                st.write(f"**Adjusted prediction (this game):** {predicted_pg:.2f}")
                if selected_prop != "anytime_td":
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
                    st.info("Use 'Anytime TD' in the player box above for TD probability.")
    else:
        st.info("Select a player to evaluate props.")

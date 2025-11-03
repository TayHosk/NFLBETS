import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import re

# ======================================================
# App config (do this once)
# ======================================================
st.set_page_config(page_title="NFL Prop & Game Model", layout="wide")

# Persistent state keys
for k, v in {
    "page": "🏈 Player Prop Model",                  # current tab
    "link_player_teams": None,                      # (team, opponent) tuple
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------------
# Sidebar Navigation
# -------------------------------
page = st.sidebar.radio(
    "Select Page:",
    ["🏈 Player Prop Model", "📈 NFL Game Predictor"],
    key="page"
)
st.sidebar.markdown("---")
st.sidebar.caption("NFL Data Model – v9.0 (Edges + Cross-link)")

# ======================================================
# Shared helpers
# ======================================================
def normalize_header(name: str) -> str:
    name = str(name).strip().replace(" ", "_").lower()
    name = re.sub(r"[^0-9a-z_]", "", name)
    return name

# ======================================================
# 🏈 TAB 1: PLAYER PROP MODEL (v7.7 + cross-link filter)
# ======================================================
if page == "🏈 Player Prop Model":
    SHEETS = {
        "total_offense": "https://docs.google.com/spreadsheets/d/1DFZRqOiMXbIoEeLaNaWh-4srxeWaXscqJxIAHt9yq48/export?format=csv",
        "total_passing": "https://docs.google.com/spreadsheets/d/1QclB5ajymBsCC09j8s4Gie_bxj4ebJwEw4kihG6uCng/export?format=csv",
        "total_rushing": "https://docs.google.com/spreadsheets/d/14NgUntobNrL1AZg3U85yZInArFkHyf9mi1csVFodu90/export?format=csv",
        "total_scoring": "https://docs.google.com/spreadsheets/d/1SJ_Y1ljU44lOjbNHuXGyKGiF3mgQxjAjX8H3j-CCqSw/export?format=csv",
        "player_receiving": "https://docs.google.com/spreadsheets/d/1Gwb2A-a4ge7UKHnC7wUpJltgioTuCQNuwOiC5ecZReM/export?format=csv",
        "player_rushing":   "https://docs.google.com/spreadsheets/d/1c0xpi_wZSf8VhkSPzzchxvhzAQHK0tFetakdRqb3e6k/export?format=csv",
        "player_passing":   "https://docs.google.com/spreadsheets/d/1I9YNSQMylW_waJs910q4S6SM8CZE--hsyNElrJeRfvk/export?format=csv",
        "def_rb":           "https://docs.google.com/spreadsheets/d/1xTP8tMnEVybu9vYuN4i6IIrI71q1j60BuqVC40fjNeY/export?format=csv",
        "def_qb":           "https://docs.google.com/spreadsheets/d/1SEwUdExz7Px61FpRNQX3bUsxVFtK97JzuQhTddVa660/export?format=csv",
        "def_wr":           "https://docs.google.com/spreadsheets/d/14klXrrHHCLlXhW6-F-9eJIz3dkp_ROXVSeehlM8TYAo/export?format=csv",
        "def_te":           "https://docs.google.com/spreadsheets/d/1yMpgtx1ObYLDVufTMR5Se3KrMi1rG6UzMzLcoptwhi4/export?format=csv",
    }

    def load_and_clean(url: str) -> pd.DataFrame:
        df = pd.read_csv(url)
        df.columns = [normalize_header(c) for c in df.columns]
        if "team" in df.columns:
            df["team"] = df["team"].astype(str).str.strip()
        elif "teams" in df.columns:
            df["team"] = df["teams"].astype(str).str.strip()
        return df

    @st.cache_data(show_spinner=False)
    def load_all():
        return {name: load_and_clean(url) for name, url in SHEETS.items()}

    data = load_all()
    p_rec, p_rush, p_pass = data["player_receiving"], data["player_rushing"], data["player_passing"]
    d_rb, d_qb, d_wr, d_te = data["def_rb"], data["def_qb"], data["def_wr"], data["def_te"]

    st.title("🏈 NFL Player Prop Model (v7.7)")

    # --- Optional filter from Game Predictor cross-link ---
    link_info = st.session_state.get("link_player_teams")
    filter_mode = st.toggle("Filter players to a specific matchup (from Game Predictor link)", value=bool(link_info))
    allowed_teams = None
    default_opponent = ""

    if filter_mode and link_info:
        team_a, team_b = link_info
        allowed_teams = {str(team_a), str(team_b)}
        default_opponent = str(team_b)

    # Build player + team lists (respect optional filter)
    if allowed_teams:
        rec_mask  = p_rec.get("team", pd.Series("")).astype(str).isin(allowed_teams)
        rush_mask = p_rush.get("team", pd.Series("")).astype(str).isin(allowed_teams)
        pass_mask = p_pass.get("team", pd.Series("")).astype(str).isin(allowed_teams)
        player_list = sorted(set(
            list(p_rec.loc[rec_mask, "player"].dropna().unique()) +
            list(p_rush.loc[rush_mask, "player"].dropna().unique()) +
            list(p_pass.loc[pass_mask, "player"].dropna().unique())
        ))
        team_list = sorted(list(allowed_teams))
    else:
        player_list = sorted(set(
            list(p_rec["player"].dropna().unique()) +
            list(p_rush["player"].dropna().unique()) +
            list(p_pass["player"].dropna().unique())
        ))
        team_list = sorted(set(
            list(d_rb["team"].dropna().unique()) +
            list(d_wr["team"].dropna().unique()) +
            list(d_te["team"].dropna().unique()) +
            list(d_qb["team"].dropna().unique())
        ))

    # --- UI ---
    player_name = st.selectbox("Select Player:", [""] + player_list)
    opponent_team = st.selectbox(
        "Select Opponent Team:",
        [""] + team_list,
        index=([""] + team_list).index(default_opponent) if default_opponent in team_list else 0
    )

    prop_choices = [
        "passing_yards", "rushing_yards", "receiving_yards",
        "receptions", "targets", "carries", "anytime_td"
    ]
    selected_props = st.multiselect("Select props:", prop_choices, default=["receiving_yards"])

    lines = {}
    for prop in selected_props:
        if prop != "anytime_td":
            lines[prop] = st.number_input(f"Sportsbook line for {prop}", value=50.0, key=f"line_{prop}")

    if not player_name or not opponent_team or not selected_props:
        st.info("Pick a player, an opponent, and at least one prop to see results.")
        st.stop()

    st.header("📊 Results")

    # --- Helper fns for props ---
    def find_player_in(df: pd.DataFrame, player_name: str):
        if "player" not in df.columns:
            return None
        mask = df["player"].astype(str).str.lower() == player_name.lower()
        return df[mask].copy() if mask.any() else None

    def detect_stat_col(df: pd.DataFrame, prop: str):
        cols = list(df.columns)
        norm = [normalize_header(c) for c in cols]
        mapping = {
            "rushing_yards": ["rushing_yards_total", "rushing_yards_per_game"],
            "receiving_yards": ["receiving_yards_total", "receiving_yards_per_game"],
            "passing_yards": ["passing_yards_total", "passing_yards_per_game"],
            "receptions":     ["receiving_receptions_total"],
            "targets":        ["receiving_targets_total"],
            "carries":        ["rushing_attempts_total", "rushing_carries_per_game"]
        }
        pri = mapping.get(prop, [])
        for cand in pri:
            if cand in norm:
                return cols[norm.index(cand)]
        return None

    def pick_def_df(prop: str, pos: str):
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

    # --- Prop logic (your v7.7 core) ---
    for prop in selected_props:
        # Anytime TD
        if prop == "anytime_td":
            st.subheader("🔥 Anytime TD (Rushing + Receiving + Defense Adjusted)")
            rec_row  = find_player_in(p_rec, player_name)
            rush_row = find_player_in(p_rush, player_name)
            total_tds, total_games = 0.0, 0.0
            for df in [rec_row, rush_row]:
                if df is not None and not df.empty:
                    td_cols = [c for c in df.columns if "td" in c and "allowed" not in c]
                    games_col = "games_played" if "games_played" in df.columns else None
                    if td_cols and games_col:
                        tds = sum(float(df.iloc[0][c]) for c in td_cols if pd.notna(df.iloc[0][c]))
                        total_tds += tds
                        total_games = max(total_games, float(df.iloc[0][games_col]))
            if total_games == 0:
                st.warning("⚠️ No games data found for this player.")
                continue
            player_td_pg = total_tds / total_games
            def_dfs = [d_rb.copy(), d_wr.copy(), d_te.copy()]
            for d in def_dfs:
                if "games_played" not in d.columns:
                    d["games_played"] = 1
                d["tds_pg"] = (
                    d[[c for c in d.columns if "td" in c and "allowed" in c]].sum(axis=1)
                    / d["games_played"].replace(0, np.nan)
                )
            league_td_pg = np.nanmean([d["tds_pg"].mean() for d in def_dfs])
            opp_td_pg = np.nanmean([
                d.loc[d["team"].astype(str).str.lower() == opponent_team.lower(), "tds_pg"].mean()
                for d in def_dfs
            ])
            if np.isnan(opp_td_pg):
                opp_td_pg = league_td_pg
            adj_factor = opp_td_pg / league_td_pg if league_td_pg > 0 else 1.0
            adj_td_rate = player_td_pg * adj_factor
            prob_anytime = min(adj_td_rate, 1.0)
            st.write(f"**Estimated Anytime TD Probability:** {prob_anytime*100:.1f}%")
            continue

        # Other props
        if prop in ["receiving_yards", "receptions", "targets"]:
            player_df, fallback_pos = find_player_in(p_rec, player_name), "wr"
        elif prop in ["rushing_yards", "carries"]:
            player_df, fallback_pos = find_player_in(p_rush, player_name), "rb"
        elif prop == "passing_yards":
            player_df, fallback_pos = find_player_in(p_pass, player_name), "qb"
        else:
            player_df = (find_player_in(p_rec, player_name)
                         or find_player_in(p_rush, player_name)
                         or find_player_in(p_pass, player_name))
            fallback_pos = "wr"

        if player_df is None or player_df.empty:
            st.warning(f"❗ {prop}: player '{player_name}' not found.")
            continue

        player_pos = player_df.iloc[0].get("position", fallback_pos)
        stat_col = detect_stat_col(player_df, prop)
        if not stat_col:
            st.warning(f"⚠️ For {prop}, no matching stat column found.")
            continue

        season_val = float(player_df.iloc[0][stat_col])
        games_played = float(player_df.iloc[0].get("games_played", 1)) or 1.0
        player_pg = season_val / games_played

        def_df = pick_def_df(prop, player_pos)
        def_col = detect_def_col(def_df, prop) if def_df is not None else None
        opp_allowed_pg = None
        league_allowed_pg = None
        if def_df is not None and def_col is not None:
            if "games_played" in def_df.columns:
                league_allowed_pg = (def_df[def_col] / def_df["games_played"].replace(0, np.nan)).mean()
            else:
                league_allowed_pg = def_df[def_col].mean()
            opp_row = def_df[def_df["team"].astype(str).str.lower() == opponent_team.lower()]
            if not opp_row.empty:
                if "games_played" in opp_row.columns and float(opp_row.iloc[0]["games_played"]) > 0:
                    opp_allowed_pg = float(opp_row.iloc[0][def_col]) / float(opp_row.iloc[0]["games_played"])
                else:
                    opp_allowed_pg = float(opp_row.iloc[0][def_col])
            else:
                opp_allowed_pg = league_allowed_pg

        adj_factor = opp_allowed_pg / league_allowed_pg if league_allowed_pg and league_allowed_pg > 0 else 1.0
        predicted_pg = player_pg * adj_factor
        line_val = lines.get(prop, 0.0)
        stdev = max(3.0, predicted_pg * 0.35)
        z = (line_val - predicted_pg) / stdev
        prob_over = 1 - norm.cdf(z)
        prob_under = norm.cdf(z)
        prob_over = float(np.clip(prob_over, 0.001, 0.999))
        prob_under = float(np.clip(prob_under, 0.001, 0.999))

        st.subheader(prop.replace("_", " ").title())
        st.write(f"**Adjusted prediction (this game):** {predicted_pg:.2f}")
        st.write(f"**Line:** {line_val}")
        st.write(f"**Probability of OVER:** {prob_over*100:.1f}%")
        st.write(f"**Probability of UNDER:** {prob_under*100:.1f}%")

        fig_bar = px.bar(
            x=["Predicted (this game)", "Line"], y=[predicted_pg, line_val],
            title=f"{player_name} – {prop.replace('_', ' ').title()}"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ======================================================
# 📈 TAB 2: NFL GAME PREDICTOR (Vegas-Calibrated + Top Edges + link to Props)
# ======================================================
elif page == "📈 NFL Game Predictor":
    st.title("📈 NFL Game Predictor (Vegas-Calibrated)")

    SCORE_URL = "https://docs.google.com/spreadsheets/d/1KrTQbR5uqlBn2v2Onpjo6qHFnLlrqIQBzE52KAhMYcY/export?format=csv"

    @st.cache_data(show_spinner=False)
    def load_scores():
        df = pd.read_csv(SCORE_URL)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    scores_df = load_scores()
    if scores_df.empty:
        st.error("❌ Could not load NFL data.")
        st.stop()

    # Inputs
    week_list = sorted(scores_df["week"].dropna().unique())
    team_list = sorted(set(scores_df["home_team"].dropna().unique()) | set(scores_df["away_team"].dropna().unique()))

    col_a, col_b = st.columns([1, 1])
    with col_a:
        selected_week = st.selectbox("Select NFL Week:", week_list, key="gp_week")
    with col_b:
        selected_team = st.selectbox("Select Your Team:", team_list, key="gp_team")

    # Helper: team averages
    def avg_scoring(df, team):
        scored_home  = df.loc[df["home_team"] == team, "home_score"].mean()
        scored_away  = df.loc[df["away_team"] == team, "away_score"].mean()
        allowed_home = df.loc[df["home_team"] == team, "away_score"].mean()
        allowed_away = df.loc[df["away_team"] == team, "home_score"].mean()
        return np.nanmean([scored_home, scored_away]), np.nanmean([allowed_home, allowed_away])

    # Calibration factor (center league to ~22.3 PPG per team)
    league_avg_pts = scores_df[["home_score", "away_score"]].stack().mean()
    cal_factor = 22.3 / league_avg_pts if not np.isnan(league_avg_pts) and league_avg_pts > 0 else 1.0

    # ------------- Section A: This-game projection + link to props -------------
    game = scores_df[
        ((scores_df["home_team"] == selected_team) | (scores_df["away_team"] == selected_team))
        & (scores_df["week"] == selected_week)
    ]

    st.markdown("### 🎯 Selected Game Projection")
    if game.empty:
        st.warning("No game data found for that team/week.")
    else:
        g = game.iloc[0]
        home, away = str(g["home_team"]), str(g["away_team"])
        opponent = away if home == selected_team else home

        # Get Vegas lines
        ou_line = float(g.get("over_under", 45.0)) if pd.notna(g.get("over_under")) else 45.0
        raw_spread = float(g.get("spread", 0.0)) if pd.notna(g.get("spread")) else 0.0

        # Normalize spread from the HOME perspective if we know favored_team
        def home_spread_from_row(row):
            if "favored_team" in row and pd.notna(row["favored_team"]) and pd.notna(row["spread"]):
                s = float(row["spread"])
                if str(row["favored_team"]).lower() == str(row["home_team"]).lower():
                    return -abs(s)
                elif str(row["favored_team"]).lower() == str(row["away_team"]).lower():
                    return abs(s)
            return float(row.get("spread", 0.0) or 0.0)

        home_spread = home_spread_from_row(g)

        # Averages
        team_avg_scored, team_avg_allowed = avg_scoring(scores_df, selected_team)
        opp_avg_scored,  opp_avg_allowed  = avg_scoring(scores_df, opponent)

        # Raw → calibrated points
        team_pts = ((team_avg_scored + opp_avg_allowed) / 2.0) * cal_factor
        opp_pts  = ((opp_avg_scored  + team_avg_allowed) / 2.0) * cal_factor

        total_pred = team_pts + opp_pts
        margin     = (team_pts - opp_pts) if selected_team == home else (opp_pts - team_pts) * -1  # margin from selected_team POV

        # Compare to Vegas
        total_diff  = total_pred - ou_line
        spread_diff = (team_pts - opp_pts) - ( -home_spread if selected_team == home else home_spread )

        c1, c2, c3 = st.columns([1.2, 1.2, 1])
        with c1:
            st.metric("Predicted Score", f"{selected_team} {team_pts:.1f} – {opponent} {opp_pts:.1f}")
        with c2:
            st.metric("Predicted Total vs O/U",
                      f"{total_pred:.1f} vs {ou_line:.1f}",
                      delta=f"{('Over' if total_diff>0 else 'Under')} {abs(total_diff):.1f} pts")
        with c3:
            st.metric("Predicted Margin vs Spread",
                      f"{(team_pts-opp_pts):.1f} vs {(-home_spread):.1f} (team POV)",
                      delta=f"{('Cover' if spread_diff>0 else 'No cover')} {abs(spread_diff):.1f} pts")

        # Charts
        fig_total = px.bar(x=["Predicted Total", "Vegas Line"], y=[total_pred, ou_line],
                           title="Predicted Total vs Vegas Line")
        st.plotly_chart(fig_total, use_container_width=True)

        fig_margin = px.bar(x=["Predicted Margin", "Vegas Spread (team POV)"], y=[(team_pts-opp_pts), (-home_spread if selected_team==home else home_spread)],
                            title="Predicted Margin vs Vegas Spread")
        st.plotly_chart(fig_margin, use_container_width=True)

        # Cross-link button → Player Props with pre-filter
        if st.button("🔗 View Player Props for This Matchup"):
            st.session_state.link_player_teams = (selected_team, opponent)
            st.session_state.page = "🏈 Player Prop Model"
            st.experimental_rerun()

    # ------------- Section B: Top Edges of the Week table -------------
    st.markdown("### 📊 Top Edges of the Week")

    dfw = scores_df[scores_df["week"] == selected_week].copy()
    if dfw.empty:
        st.info("No games found for this week.")
    else:
        rows = []
        for _, r in dfw.iterrows():
            try:
                home, away = str(r["home_team"]), str(r["away_team"])
                # Team averages season-to-date
                h_scored, h_allowed = avg_scoring(scores_df, home)
                a_scored, a_allowed = avg_scoring(scores_df, away)
                # Model points (calibrated)
                home_pts = ((h_scored + a_allowed) / 2.0) * cal_factor
                away_pts = ((a_scored + h_allowed) / 2.0) * cal_factor
                model_total  = home_pts + away_pts
                model_margin = home_pts - away_pts  # home perspective

                # Vegas lines
                ou_line = float(r.get("over_under", np.nan)) if pd.notna(r.get("over_under")) else np.nan
                # Normalize spread to home perspective
                if "favored_team" in r and pd.notna(r["favored_team"]) and pd.notna(r["spread"]):
                    s = float(r["spread"])
                    if str(r["favored_team"]).lower() == home.lower():
                        home_spread = -abs(s)
                    elif str(r["favored_team"]).lower() == away.lower():
                        home_spread = abs(s)
                    else:
                        home_spread = float(r.get("spread", 0.0) or 0.0)
                else:
                    home_spread = float(r.get("spread", 0.0) or 0.0)

                # Edges
                total_edge  = model_total - ou_line if not np.isnan(ou_line) else np.nan
                spread_edge = model_margin - home_spread

                rows.append({
                    "week": selected_week,
                    "matchup": f"{away} @ {home}",
                    "model_home_pts": round(home_pts, 1),
                    "model_away_pts": round(away_pts, 1),
                    "model_total": round(model_total, 1),
                    "vegas_ou": ou_line if not np.isnan(ou_line) else None,
                    "total_edge_pts": None if np.isnan(total_edge) else round(total_edge, 1),
                    "home_spread": round(home_spread, 1),
                    "spread_edge_pts": round(spread_edge, 1),
                    "edge_flag_total": "Over" if (not np.isnan(total_edge) and total_edge > 0) else ("Under" if (not np.isnan(total_edge) and total_edge < 0) else ""),
                    "edge_flag_spread": "Home cover" if spread_edge > 0 else "Away cover"
                })
            except Exception:
                # Robust to partial rows
                continue

        if rows:
            edges_df = pd.DataFrame(rows)
            # Sort by absolute best edge (pick which metric to prioritize)
            sort_choice = st.selectbox("Sort edges by:", ["Total edge (abs)", "Spread edge (abs)"])
            if sort_choice.startswith("Total"):
                edges_df = edges_df.sort_values(by=edges_df["total_edge_pts"].abs().fillna(0), ascending=False)
            else:
                edges_df = edges_df.sort_values(by=edges_df["spread_edge_pts"].abs(), ascending=False)

            top_n = st.slider("Show top N", min_value=5, max_value=len(edges_df), value=min(10, len(edges_df)))
            st.dataframe(edges_df.head(top_n), use_container_width=True)

            csv = edges_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download full edges CSV", data=csv, file_name=f"week_{selected_week}_edges.csv", mime="text/csv")
        else:
            st.info("No edge rows could be computed for this week.")

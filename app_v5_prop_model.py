import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import re

# ======================================================
# App config
# ======================================================
st.set_page_config(page_title="NFL Prop & Game Model", layout="wide")

# Initialize session state
for k, v in {
    "page": "🏈 Player Prop Model",
    "link_player_teams": None,
    "next_page": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# Sidebar Navigation (Option 1 safe switching)
# ======================================================
if st.session_state.next_page:
    default_index = ["🏈 Player Prop Model", "📈 NFL Game Predictor"].index(st.session_state.next_page)
    st.session_state.next_page = None
else:
    default_index = 0

page = st.sidebar.radio(
    "Select Page:",
    ["🏈 Player Prop Model", "📈 NFL Game Predictor"],
    index=default_index,
    key="page"
)
st.sidebar.markdown("---")
st.sidebar.caption("NFL Data Model – v9.1 (Edges + Cross-Link Fix)")

# ======================================================
# Shared helper
# ======================================================
def normalize_header(name: str) -> str:
    name = str(name).strip().replace(" ", "_").lower()
    name = re.sub(r"[^0-9a-z_]", "", name)
    return name


# ======================================================
# 🏈 PLAYER PROP MODEL
# ======================================================
if page == "🏈 Player Prop Model":
    SHEETS = {
        "player_receiving": "https://docs.google.com/spreadsheets/d/1Gwb2A-a4ge7UKHnC7wUpJltgioTuCQNuwOiC5ecZReM/export?format=csv",
        "player_rushing":   "https://docs.google.com/spreadsheets/d/1c0xpi_wZSf8VhkSPzzchxvhzAQHK0tFetakdRqb3e6k/export?format=csv",
        "player_passing":   "https://docs.google.com/spreadsheets/d/1I9YNSQMylW_waJs910q4S6SM8CZE--hsyNElrJeRfvk/export?format=csv",
        "def_rb":           "https://docs.google.com/spreadsheets/d/1xTP8tMnEVybu9vYuN4i6IIrI71q1j60BuqVC40fjNeY/export?format=csv",
        "def_qb":           "https://docs.google.com/spreadsheets/d/1SEwUdExz7Px61FpRNQX3bUsxVFtK97JzuQhTddVa660/export?format=csv",
        "def_wr":           "https://docs.google.com/spreadsheets/d/14klXrrHHCLlXhW6-F-9eJIz3dkp_ROXVSeehlM8TYAo/export?format=csv",
        "def_te":           "https://docs.google.com/spreadsheets/d/1yMpgtx1ObYLDVufTMR5Se3KrMi1rG6UzMzLcoptwhi4/export?format=csv",
    }

    def load_and_clean(url):
        df = pd.read_csv(url)
        df.columns = [normalize_header(c) for c in df.columns]
        if "team" in df.columns:
            df["team"] = df["team"].astype(str).str.strip()
        return df

    @st.cache_data(show_spinner=False)
    def load_all():
        return {n: load_and_clean(u) for n, u in SHEETS.items()}

    data = load_all()
    p_rec, p_rush, p_pass = data["player_receiving"], data["player_rushing"], data["player_passing"]
    d_rb, d_qb, d_wr, d_te = data["def_rb"], data["def_qb"], data["def_wr"], data["def_te"]

    st.title("🏈 NFL Player Prop Model (v7.7)")

    # --- cross-link filter from Game Predictor ---
    link = st.session_state.get("link_player_teams")
    filter_mode = st.toggle("Filter players by linked matchup", value=bool(link))
    allowed, default_opponent = None, ""
    if filter_mode and link:
        t1, t2 = link
        allowed, default_opponent = {str(t1), str(t2)}, str(t2)

    # --- lists ---
    if allowed:
        mask = lambda df: df.get("team", pd.Series("")).astype(str).isin(allowed)
        player_list = sorted(set(
            list(p_rec.loc[mask(p_rec), "player"].dropna()) +
            list(p_rush.loc[mask(p_rush), "player"].dropna()) +
            list(p_pass.loc[mask(p_pass), "player"].dropna())
        ))
        team_list = sorted(list(allowed))
    else:
        player_list = sorted(set(
            list(p_rec["player"].dropna()) +
            list(p_rush["player"].dropna()) +
            list(p_pass["player"].dropna())
        ))
        team_list = sorted(set(
            list(d_rb["team"].dropna()) +
            list(d_wr["team"].dropna()) +
            list(d_te["team"].dropna()) +
            list(d_qb["team"].dropna())
        ))

    # --- inputs ---
    player = st.selectbox("Select Player:", [""] + player_list)
    opp = st.selectbox("Select Opponent Team:", [""] + team_list,
                       index=([""] + team_list).index(default_opponent)
                       if default_opponent in team_list else 0)

    props = st.multiselect("Select props:",
        ["passing_yards","rushing_yards","receiving_yards","receptions","targets","carries","anytime_td"],
        default=["receiving_yards"]
    )

    lines = {p: st.number_input(f"Line for {p}", value=50.0, key=f"line_{p}") for p in props if p!="anytime_td"}

    if not player or not opp or not props:
        st.info("Pick a player, opponent, and at least one prop.")
        st.stop()

    # --- helpers ---
    def find(df, name):
        if "player" not in df.columns: return None
        m = df["player"].astype(str).str.lower() == name.lower()
        return df[m].copy() if m.any() else None

    def stat_col(df, prop):
        cols = list(df.columns); norm = [normalize_header(c) for c in cols]
        map_ = {
            "rushing_yards":["rushing_yards_total","rushing_yards_per_game"],
            "receiving_yards":["receiving_yards_total","receiving_yards_per_game"],
            "passing_yards":["passing_yards_total","passing_yards_per_game"],
            "receptions":["receiving_receptions_total"],"targets":["receiving_targets_total"],
            "carries":["rushing_attempts_total","rushing_carries_per_game"]
        }
        for c in map_.get(prop, []):
            if c in norm: return cols[norm.index(c)]
        return None

    def def_df(prop,pos):
        if prop=="passing_yards": return d_qb
        if prop in ["rushing_yards","carries"]: return d_rb if pos!="qb" else d_qb
        if prop in ["receiving_yards","receptions","targets"]:
            if pos=="te": return d_te
            if pos=="rb": return d_rb
            return d_wr
        return None

    def def_col(df,prop):
        cols=list(df.columns); norm=[normalize_header(c) for c in cols]
        pref={
            "rushing_yards":["rushing_yards_allowed_total","rushing_yards_allowed"],
            "receiving_yards":["receiving_yards_allowed_total","receiving_yards_allowed"],
            "passing_yards":["passing_yards_allowed_total","passing_yards_allowed"]
        }.get(prop,[])
        for c in pref:
            if c in norm: return cols[norm.index(c)]
        for i,c in enumerate(norm):
            if "allowed" in c: return cols[i]
        return None

    # --- compute ---
    st.header("📊 Results")
    for p in props:
        if p=="anytime_td":
            st.subheader("🔥 Anytime TD Probability")
            rec, rush = find(p_rec, player), find(p_rush, player)
            tds,g=0,0
            for df in [rec,rush]:
                if df is not None and not df.empty:
                    tcols=[c for c in df if "td" in c and "allowed" not in c]
                    gcol="games_played" if "games_played" in df.columns else None
                    if tcols and gcol:
                        tds+=sum(float(df.iloc[0][c]) for c in tcols if pd.notna(df.iloc[0][c]))
                        g=max(g,float(df.iloc[0][gcol]))
            if g==0: st.warning("No data."); continue
            rate=tds/g
            st.write(f"**TDs/game ≈ {rate:.2f}** → **Anytime TD prob {min(rate,1)*100:.1f}%**")
            continue

        if p in ["receiving_yards","receptions","targets"]:
            pdf,pos=find(p_rec,player),"wr"
        elif p in ["rushing_yards","carries"]:
            pdf,pos=find(p_rush,player),"rb"
        elif p=="passing_yards":
            pdf,pos=find(p_pass,player),"qb"
        else:
            pdf,pos=find(p_rec,player),"wr"
        if pdf is None or pdf.empty:
            st.warning(f"No data for {player}"); continue
        col=stat_col(pdf,p)
        s=float(pdf.iloc[0][col]); g=float(pdf.iloc[0].get("games_played",1))
        pg=s/g
        ddf=def_df(p,pos); dcol=def_col(ddf,p)
        lg=(ddf[dcol]/ddf.get("games_played",1)).mean()
        orow=ddf[ddf["team"].str.lower()==opp.lower()]
        op=float(orow.iloc[0][dcol])/float(orow.iloc[0].get("games_played",1))
        pred=pg*(op/lg)
        line=lines.get(p,50.0)
        sd=max(3.0,pred*0.35); z=(line-pred)/sd
        over=1-norm.cdf(z)
        st.subheader(p.replace("_"," ").title())
        st.write(f"Pred {pred:.1f} vs Line {line:.1f} → Over {over*100:.1f}%")
        st.plotly_chart(px.bar(x=["Pred","Line"],y=[pred,line],title=p),use_container_width=True)


# ======================================================
# 📈 GAME PREDICTOR
# ======================================================
elif page == "📈 NFL Game Predictor":
    st.title("📈 NFL Game Predictor (Vegas-Calibrated)")
    URL="https://docs.google.com/spreadsheets/d/1KrTQbR5uqlBn2v2Onpjo6qHFnLlrqIQBzE52KAhMYcY/export?format=csv"

    @st.cache_data(show_spinner=False)
    def load_scores():
        df=pd.read_csv(URL); df.columns=[c.strip().lower().replace(" ","_") for c in df.columns]; return df
    df=load_scores()
    weeks=sorted(df["week"].dropna().unique())
    teams=sorted(set(df["home_team"].dropna())|set(df["away_team"].dropna()))
    w=st.selectbox("Week:",weeks); t=st.selectbox("Team:",teams)

    def avg(df,team):
        sh=df.loc[df["home_team"]==team,"home_score"].mean()
        sa=df.loc[df["away_team"]==team,"away_score"].mean()
        ah=df.loc[df["home_team"]==team,"away_score"].mean()
        aa=df.loc[df["away_team"]==team,"home_score"].mean()
        return np.nanmean([sh,sa]),np.nanmean([ah,aa])

    lg=df[["home_score","away_score"]].stack().mean(); cal=22.3/lg if lg>0 else 1
    g=df[((df["home_team"]==t)|(df["away_team"]==t))&(df["week"]==w)]
    if not g.empty:
        g=g.iloc[0]; o=g["away_team"] if g["home_team"]==t else g["home_team"]
        ts,ta=avg(df,t); os,oa=avg(df,o)
        tp,op=((ts+oa)/2)*cal,((os+ta)/2)*cal
        ou=st.number_input("Over/Under:",value=float(g.get("over_under",45)))
        sp=st.number_input("Spread:",value=float(g.get("spread",0)))
        tot=tp+op; diff=tot-ou; spr=tp-op-(-sp)
        st.write(f"**{t}: {tp:.1f} | {o}: {op:.1f} | Total: {tot:.1f} | Lean:** {'Over' if diff>0 else 'Under'}")
        if st.button("🔗 View Player Props for This Matchup"):
            st.session_state.link_player_teams=(t,o)
            st.session_state.next_page="🏈 Player Prop Model"
            st.experimental_rerun()

    # --- Top Edges ---
    st.markdown("### 📊 Top Edges of the Week")
    sub=df[df["week"]==w]; rows=[]
    for _,r in sub.iterrows():
        h,a=r["home_team"],r["away_team"]
        hs,ha=avg(df,h); as_,aa=avg(df,a)
        hp,ap=((hs+aa)/2)*cal,((as_+ha)/2)*cal
        ou=float(r.get("over_under",45)); sp=float(r.get("spread",0))
        rows.append({
            "matchup":f"{a}@{h}","model_total":round(hp+ap,1),"vegas_ou":ou,
            "edge_total":round((hp+ap)-ou,1),
            "model_margin":round(hp-ap,1),"spread":sp,
            "edge_spread":round((hp-ap)-(-sp),1)
        })
    tab=pd.DataFrame(rows).sort_values("edge_total",key=lambda s:s.abs(),ascending=False)
    st.dataframe(tab.head(10),use_container_width=True)

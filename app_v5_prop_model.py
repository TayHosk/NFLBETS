
# app_v5_prop_model.py
# NFL Player Prop Model – Google Sheets version (Taylor)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import matplotlib.pyplot as plt

st.set_page_config(page_title="NFL Player Prop Model (Sheets v5)", layout="centered")

# Google Sheets URLs
SHEET_TOTAL_OFFENSE = "https://docs.google.com/spreadsheets/d/1DFZRqOiMXbIoEeLaNaWh-4srxeWaXscqJxIAHt9yq48/export?format=csv"
SHEET_TOTAL_PASS_OFF = "https://docs.google.com/spreadsheets/d/1QclB5ajymBsCC09j8s4Gie_bxj4ebJwEw4kihG6uCng/export?format=csv"
SHEET_TOTAL_RUSH_OFF = "https://docs.google.com/spreadsheets/d/14NgUntobNrL1AZg3U85yZInArFkHyf9mi1csVFodu90/export?format=csv"
SHEET_TOTAL_SCORE_OFF = "https://docs.google.com/spreadsheets/d/1SJ_Y1ljU44lOjbNHuXGyKGiF3mgQxjAjX8H3j-CCqSw/export?format=csv"
SHEET_PLAYER_REC = "https://docs.google.com/spreadsheets/d/1Gwb2A-a4ge7UKHnC7wUpJltgioTuCQNuwOiC5ecZReM/export?format=csv"
SHEET_PLAYER_RUSH = "https://docs.google.com/spreadsheets/d/1c0xpi_wZSf8VhkSPzzchxvhzAQHK0tFetakdRqb3e6k/export?format=csv"
SHEET_PLAYER_PASS = "https://docs.google.com/spreadsheets/d/1I9YNSQMylW_waJs910q4S6SM8CZE--hsyNElrJeRfvk/export?format=csv"
SHEET_DEF_RB = "https://docs.google.com/spreadsheets/d/1xTP8tMnEVybu9vYuN4i6IIrI71q1j60BuqVC40fjNeY/export?format=csv"
SHEET_DEF_QB = "https://docs.google.com/spreadsheets/d/1SEwUdExz7Px61FpRNQX3bUsxVFtK97JzuQhTddVa660/export?format=csv"
SHEET_DEF_WR = "https://docs.google.com/spreadsheets/d/14klXrrHHCLlXhW6-F-9eJIz3dkp_ROXVSeehlM8TYAo/export?format=csv"
SHEET_DEF_TE = "https://docs.google.com/spreadsheets/d/1yMpgtx1ObYLDVufTMR5Se3KrMi1rG6UzMzLcoptwhi4/export?format=csv"

def load_sheet(url): return pd.read_csv(url)

@st.cache_data
def load_all_data():
    try:
        return {
            "total_off": load_sheet(SHEET_TOTAL_OFFENSE),
            "total_pass_off": load_sheet(SHEET_TOTAL_PASS_OFF),
            "total_rush_off": load_sheet(SHEET_TOTAL_RUSH_OFF),
            "total_score_off": load_sheet(SHEET_TOTAL_SCORE_OFF),
            "p_rec": load_sheet(SHEET_PLAYER_REC),
            "p_rush": load_sheet(SHEET_PLAYER_RUSH),
            "p_pass": load_sheet(SHEET_PLAYER_PASS),
            "d_rb": load_sheet(SHEET_DEF_RB),
            "d_qb": load_sheet(SHEET_DEF_QB),
            "d_wr": load_sheet(SHEET_DEF_WR),
            "d_te": load_sheet(SHEET_DEF_TE),
        }
    except Exception as e:
        st.error(f"Error loading Google Sheets: {e}")
        st.stop()

data = load_all_data()

st.title("🏈 NFL Player Prop Model (Google Sheets v5)")
st.write("Estimate probabilities for NFL player props using Google Sheets data.")

player_name = st.text_input("Player name:")
opponent_team = st.text_input("Opponent team:")
prop = st.selectbox("Prop type", ["passing_yards", "rushing_yards", "receiving_yards"])
line_value = st.number_input("Sportsbook line", value=50.0)

if player_name and opponent_team:
    st.write(f"Analyzing {player_name} vs {opponent_team} for {prop}...")
    st.write("✅ Data loaded and ready for modeling!")
else:
    st.info("Enter player name and opponent to begin.")

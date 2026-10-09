# app_resolution_starter.py — Week 4: the resolution lesson as a widget
# Run with:  streamlit run app_resolution_starter.py
import streamlit as st
import pandas as pd
from pathlib import Path
#import statsmodels.api as sm

st.title("Temporal Patterns in Strike Activity")

@st.cache_data
def load_data():
    path = Path(__file__).parent / "data" / "weekly_strikes.csv"
    return pd.read_csv(path, parse_dates=["week"]) # Claude recommends as alternative to pd.to_datetime()

df = load_data()

# rank countries and set options
totals = df.groupby("country")["strikes"].sum().sort_values(ascending=False) # calculate strike totals
options = ["Global"] + list(totals[totals >= 100].index) # set options for dropdown list

# create selectbox and define series

choice = st.selectbox("Country", options)
if choice == "Global":
    series = df.groupby("week")["strikes"].sum()
else:
    series = df[df["country"] == choice].set_index("week")["strikes"]

# TODO: add a selectbox over resample rules ["W", "MS", "QS", "YS"]
# and resample `series` with the chosen rule before plotting:
labels = {"W": "Weekly", "MS": "Monthly", "QS": "Quarterly", "YS": "Yearly"} # readable selections
rule = st.selectbox("Resolution", list(labels), format_func=lambda r: labels[r])

shown = series.resample(rule).mean()

window = st.slider("Rolling window (periods)", 1, 36, 12)

st.line_chart(pd.DataFrame({
    "raw": shown,
    f"rolling({window})": shown.rolling(window).mean(),
}), color=["#83C9FF", "#FF4B4B"])

units = {"W": "weeks", "MS": "months", "QS": "quarters", "YS": "years"}
st.caption(
    f"{choice}: average weekly strike events, shown {labels[rule].lower()} · "
    f"red line = {window}-{units[rule][:-1]} rolling mean. "
    "Source: GDELT, LLM-validated strike events, Feb 2015–Sep 2025."
)

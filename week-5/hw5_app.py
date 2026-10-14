# hw5_app.py — Week 5: the Week 4 strike time series, linked to an animated ADM-1 map
# Run with:  streamlit run week-5/hw5_app.py
# Data built by wrangle_adm1.py (run that first if week-5/data/ is empty).
import json
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA = Path(__file__).parent / "data"

st.set_page_config(page_title="Strikes by Region", layout="wide")
st.title("Where and When Strikes Happen")


# --- Data --------------------------------------------------------------------
@st.cache_data
def load_weekly():
    return pd.read_csv(DATA / "weekly_strikes.csv", parse_dates=["week"])


@st.cache_data
def load_adm1():
    return pd.read_csv(DATA / "adm1_monthly.csv", parse_dates=["month"])


@st.cache_data
def load_boundaries():
    return json.loads((DATA / "adm1.geojson").read_text())


@st.cache_data
def load_centers():
    return pd.read_csv(DATA / "country_centers.csv", index_col="iso3")


@st.cache_data
def load_map_lines():
    """Extra lines per country (for now, Kashmir's Line of Control) as
    {iso3: [(lon, lat, dash), ...]}, with None between the pieces of a line."""
    lines = {}
    for f in json.loads((DATA / "map_lines.geojson").read_text())["features"]:
        geom = f["geometry"]
        parts = [geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]
        lon = [x for part in parts for x in [p[0] for p in part] + [None]]
        lat = [y for part in parts for y in [p[1] for p in part] + [None]]
        lines.setdefault(f["properties"]["iso3"], []).append((lon, lat, f["properties"]["dash"]))
    return lines


@st.cache_data
def load_dots():
    """Where to put a dot for regions too small to see (Moscow City, DC), by gid_1."""
    return pd.read_csv(DATA / "region_dots.csv", index_col="gid_1")


weekly, adm1, boundaries, centers = load_weekly(), load_adm1(), load_boundaries(), load_centers()

# rank countries and set options (same rule as Week 4)
totals = weekly.groupby("country")["strikes"].sum().sort_values(ascending=False)
options = ["Global"] + list(totals[totals >= 100].index)
iso3_of = weekly.drop_duplicates("country").set_index("country")["iso3"]

# --- Controls ----------------------------------------------------------------
# One resolution and one date range drive both panels, so the map and the line
# always show the same periods. Weekly is left out: most regions have no strike
# in a given week, so a weekly map is mostly blank.
labels = {"MS": "Monthly", "QS": "Quarterly", "YS": "Yearly"}
months = list(adm1["month"].drop_duplicates().sort_values())

with st.sidebar:
    choice = st.selectbox("Country", options, index=1)  # default to the top country
    rule = st.radio("Resolution", list(labels), format_func=lambda r: labels[r],
                    help="Each map frame and each point on the line is one period's total.")
    start, end = st.select_slider("Period", options=months, value=(months[0], months[-1]),
                                  format_func=lambda t: t.strftime("%b %Y"))
    window = st.slider("Rolling window (periods)", 1, 36, 3)

    st.divider()
    with st.expander("About this app"):
        st.markdown("""
**Data.** GDELT strike events, Feb 2015–Sep 2025, each confirmed by an LLM
classifier. Counts are events, not strikers.

**Map vs. line.** The line counts all of a country's strikes. The map shows only
those GDELT places in a region; strikes coded to the country as a whole are left off.

**Dots** mark regions too small to see, such as Moscow City or Washington, DC.

**Color scale** is capped at the 95th percentile, so one huge month doesn't wash
out the rest.
""")


end_excl = end + pd.offsets.MonthBegin(1)  # the slider's end month counts in full
if choice == "Global":
    iso3 = None
    series = weekly.groupby("week")["strikes"].sum()
else:
    iso3 = iso3_of[choice]
    series = weekly[weekly["iso3"] == iso3].set_index("week")["strikes"]
series = series[(series.index >= start) & (series.index < end_excl)]
regions = adm1[(adm1["iso3"] == iso3) & adm1["month"].between(start, end)]

# --- Headline numbers --------------------------------------------------------
on_map = regions["strikes"].sum()

c1, c2, c3 = st.columns(3)
c1.metric("Strike events", f"{series.sum():,}")
if len(regions):
    c2.metric("Located in a region", f"{on_map:,}",
              f"{on_map / max(series.sum(), 1):.0%} of the country's strikes", delta_color="off")
    top = regions.groupby("region")["strikes"].sum().sort_values(ascending=False)
    c3.metric("Most strikes", top.index[0], f"{top.iloc[0]:,} events", delta_color="off")


# --- Figure ------------------------------------------------------------------
# Map and line are one plotly figure, so one play button drives both: each frame
# recolors the map and moves a marker along the line. Plotly's animation_frame
# would copy the boundaries into every frame; here they go in once and each frame
# carries only what changes, so playback runs in the browser without re-running
# this script.
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FRAME_MS = {"MS": 250, "QS": 600, "YS": 1200}
MAP_PAD = 0.5  # degrees of margin around each country
UNIT = {"MS": "month", "QS": "quarter", "YS": "year"}


def period_label(t, rule):
    return {"MS": t.strftime("%b %Y"), "QS": f"{t.year} Q{t.quarter}", "YS": str(t.year)}[rule]


def line_traces(shown, window):
    return [
        go.Scatter(x=shown.index, y=shown.values, name="strikes", mode="lines",
                   line=dict(color="#83C9FF", width=1.5),
                   hovertemplate="%{x|%b %Y}: %{y} strikes<extra></extra>"),
        go.Scatter(x=shown.index, y=shown.rolling(window).mean().values,
                   name=f"rolling({window})", mode="lines", line=dict(color="#FF4B4B", width=2),
                   hovertemplate="%{x|%b %Y}: %{y:.1f}<extra>rolling mean</extra>"),
    ]


def style_line_axes(fig, rule, **where):
    fig.update_xaxes(showgrid=False, **where)
    fig.update_yaxes(title_text=f"Strikes per {UNIT[rule]}", rangemode="tozero",
                     gridcolor="rgba(128,128,128,0.2)", **where)


@st.cache_data
def line_figure(series, rule, window):
    """The line on its own, for Global and for countries with no regional data."""
    fig = go.Figure(line_traces(series.resample(rule).sum(), window))
    style_line_axes(fig, rule)
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
    return fig


@st.cache_data
def linked_figure(iso3, series, rule, start, end, window):
    wide = (adm1[(adm1["iso3"] == iso3) & adm1["month"].between(start, end)]
            .pivot(index="month", columns="gid_1", values="strikes")
            .resample(rule).sum())
    shown = series.resample(rule).sum().reindex(wide.index, fill_value=0)
    names = adm1.drop_duplicates("gid_1").set_index("gid_1")["region"].reindex(wide.columns)
    ids = set(wide.columns)
    geo = {"type": "FeatureCollection",
           "features": [f for f in boundaries["features"] if f["properties"]["gid_1"] in ids]}

    # A few region-periods dwarf the rest (a general strike in one state), so the
    # scale tops out at the 95th percentile of non-zero cells; anything above is darkest.
    nonzero = wide.values[wide.values > 0]
    zmax = max(math.ceil(pd.Series(nonzero).quantile(0.95)), 2) if len(nonzero) else 1
    ticks = sorted({0, zmax // 2, zmax})
    ytop = 1.05 * max(shown.max(), 1)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.14,
                        specs=[[{"type": "geo"}, {"type": "xy"}]])
    # trace 0: the map
    fig.add_trace(go.Choropleth(
        geojson=geo, featureidkey="properties.gid_1",
        locations=list(wide.columns), z=wide.iloc[0].values, text=names.values,
        zmin=0, zmax=zmax, colorscale=BLUES,
        marker_line_color="rgba(128,128,128,0.7)", marker_line_width=0.6,
        hovertemplate="<b>%{text}</b><br>%{z} strikes<extra></extra>",
        colorbar=dict(title="Strikes", thickness=12, len=0.6, x=0.48, xanchor="left",
                      tickvals=ticks, ticktext=[str(t) for t in ticks[:-1]] + [f"{zmax}+"]),
    ), row=1, col=1)
    # traces 1-2: the line and its rolling mean; trace 3: where the map is now
    for trace in line_traces(shown, window):
        fig.add_trace(trace, row=1, col=2)
    now = wide.index[0]
    fig.add_trace(go.Scatter(x=[now, now], y=[0, ytop], mode="lines", showlegend=False,
                             line=dict(color="rgba(128,128,128,0.8)", width=1.5, dash="dot"),
                             hoverinfo="skip"), row=1, col=2)
    # trace 4, if any: dots for regions too small to see, on the same color scale
    dots = load_dots()
    dots = dots[dots.index.isin(wide.columns)]
    if len(dots):
        fig.add_trace(go.Scattergeo(
            lon=dots["lon"], lat=dots["lat"], text=names[dots.index].values,
            mode="markers", showlegend=False,
            marker=dict(size=11, color=wide[dots.index].iloc[0].values, cmin=0, cmax=zmax,
                        colorscale=BLUES, line=dict(color="#555555", width=1)),
            hovertemplate="<b>%{text}</b><br>%{marker.color} strikes<extra></extra>",
        ), row=1, col=1)
    style_line_axes(fig, rule, row=1, col=2)
    fig.update_yaxes(range=[0, ytop], row=1, col=2)

    # Extra lines: India's Line of Control, dotted as on UN maps (Jammu and Kashmir
    # is colored like any region but labeled disputed).
    for lon, lat, dash in load_map_lines().get(iso3, []):
        fig.add_trace(go.Scattergeo(lon=lon, lat=lat, mode="lines", showlegend=False,
                                    line=dict(color="#777777", width=1.2, dash=dash),
                                    hoverinfo="skip"), row=1, col=1)

    frames = [period_label(t, rule) for t in wide.index]
    fig.frames = [go.Frame(data=[go.Choropleth(z=row.values), go.Scatter(x=[t, t])]
                           + ([go.Scattergeo(marker=dict(color=row[dots.index].values))]
                              if len(dots) else []),
                           traces=[0, 3, 4] if len(dots) else [0, 3], name=label)
                  for label, (t, row) in zip(frames, wide.iterrows())]

    # An equal-area projection centered on the country keeps Canada and Russia from
    # being stretched east-west. The US gets Albers USA, which insets Alaska and
    # Hawaii instead of shrinking the lower 48 to fit them.
    if iso3 == "USA":
        fig.update_geos(scope="usa", projection_type="albers usa")
    else:
        # fitbounds doesn't zoom azimuthal projections properly. Limiting the map to
        # the country's lon/lat box does: plotly fits that box to the panel.
        c = centers.loc[iso3]
        fig.update_geos(projection_type="azimuthal equal area",
                        projection_rotation=dict(lon=c["lon"], lat=c["lat"]),
                        lonaxis_range=[c["lon_min"] - MAP_PAD, c["lon_max"] + MAP_PAD],
                        lataxis_range=[c["lat_min"] - MAP_PAD, c["lat_max"] + MAP_PAD],
                        # without this, plotly centers on the middle of the lon/lat
                        # box, which is not the rotation point, and the map slides
                        # off the frame (Russia's far east was cut off)
                        center=dict(lon=c["lon"], lat=c["lat"]))
    fig.update_geos(visible=False, bgcolor="rgba(0,0,0,0)")

    step = dict(frame=dict(duration=FRAME_MS[rule], redraw=True), transition=dict(duration=0))
    fig.update_layout(
        height=520, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", x=0.6, y=1.02, yanchor="bottom"),
        updatemenus=[dict(
            type="buttons", direction="left", x=0, y=0, xanchor="left", yanchor="top",
            pad=dict(t=40, r=10), showactive=False,
            buttons=[dict(label="▶ Play", method="animate", args=[None, {**step, "fromcurrent": True}]),
                     dict(label="❚❚ Pause", method="animate",
                          args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])],
        )],
        sliders=[dict(
            x=0.16, y=0, len=0.84, xanchor="left", yanchor="top", pad=dict(t=30),
            currentvalue=dict(prefix="", font=dict(size=16)),
            steps=[dict(label=f, method="animate",
                        args=[[f], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                   for f in frames],
        )],
    )
    return fig


# --- Layout ------------------------------------------------------------------
# General notes live in "About this app"; under the map go only notes about the
# country on screen.
NOTES = {
    "IND": "Jammu and Kashmir is labeled as disputed, as on UN maps: the dotted line marks "
           "the approximate Line of Control between India and Pakistan, and the region's "
           "final status has not been agreed by the parties.",
    "ESP": "The Canary Islands are left off the map so that mainland Spain isn't shrunk "
           "into one corner. Their strikes are still counted in the line.",
    "BEL": "The GDELT-to-GADM crosswalk has no Belgian matches, so Belgium's strikes are "
           "placed in its three regions by their coordinates.",
    "USA": "Alaska and Hawaii are shown as insets.",
}

if iso3 is not None and not regions.empty:
    st.plotly_chart(linked_figure(iso3, series, rule, start, end, window))
    if iso3 in NOTES:
        st.caption(NOTES[iso3])
else:
    if choice == "Global":
        st.info("Pick a country in the sidebar to see its first-level regions "
                "(states, provinces) on a map alongside the line.")
    else:
        st.info(f"No regional (ADM-1) data for {choice}, so only the time series is shown.")
    st.plotly_chart(line_figure(series, rule, window))

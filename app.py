"""DESDR Bangladesh flood-year survey — local interactive geomap.

Run:  streamlit run app.py
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

import processing as P

st.set_page_config(page_title="DESDR Flood Survey Map", page_icon="", layout="wide")

# Ordinal one-hue ramp, rank 1 (worst flood year) darkest. Validated with the
# dataviz palette validator (--ordinal, light surface).
RANK_COLORS = {1: "#104281", 2: "#1c5cab", 3: "#2a78d6", 4: "#5598e7", 5: "#86b6ef"}
RANK_LABELS = {1: "1 · worst", 2: "2", 3: "3", 4: "4", 5: "5 · fifth worst"}
NO_DATA = "#898781"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
ALL = "All locations"


@st.cache_data(show_spinner=False)
def load(path: str, mtime: float) -> P.SurveyData:  # mtime busts the cache when the file changes
    return P.load_dataset(path)


@st.cache_data(show_spinner=False)
def load_boundaries() -> dict | None:
    if P.BOUNDARIES_GEOJSON.exists():
        return json.loads(P.BOUNDARIES_GEOJSON.read_text())
    return None


def rank_color(value: float) -> str:
    return RANK_COLORS[int(min(5, max(1, round(value))))]


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("Controls")
# Fixed path: a free-text path box would let any visitor of the hosted app read files on the server.
path = str(P.DEFAULT_PARQUET)
if not Path(path).exists():
    st.error(f"Survey file not found: `{Path(path).name}`")
    st.stop()
try:
    data = load(path, Path(path).stat().st_mtime)
except ValueError as e:
    st.error(f"This file doesn't look like the flood-ranking survey: {e}")
    st.stop()

gaz, clean = data.gazetteer, data.clean
year = st.sidebar.slider("Flood year", P.YEAR_MIN, P.YEAR_MAX, 2004 if 2004 in clean["year"].values else P.YEAR_MIN)
if "location" not in st.session_state:
    st.session_state.location = ALL
if "_clicked_location" in st.session_state:  # set by a map click on the previous run
    st.session_state.location = st.session_state.pop("_clicked_location")
location = st.sidebar.selectbox("Location", [ALL, *gaz["survey_name"]], key="location",
                                help="Also set by clicking a marker on the map.")
metric = st.sidebar.radio("Colour markers by", ["Mean rank", "Median rank"], horizontal=True)
metric_col = "mean_rank" if metric == "Mean rank" else "median_rank"
show_bounds = st.sidebar.checkbox("Show upazila boundaries", value=True)

# One filtered record set feeds the map, the charts and the tables below.
view = clean if location == ALL else clean[clean["location"] == location]
stats = P.year_location_stats(clean, gaz, year)

st.sidebar.markdown("---")
st.sidebar.caption(f"Validated records: **{len(clean)}** · rejected: **{len(data.issues)}** · "
                   f"respondents: **{clean['respondent'].nunique()}**")

# --------------------------------------------------------------------------- header
st.title("Bangladesh flood-year survey")
st.caption("Respondents rank the five worst flood years 2000–2020 for their upazila. "
           "**Rank 1 = worst flood year.** Ranks are ordinal opinions, not flood-intensity measurements; "
           "a year missing from someone's top five is *unranked*, not 'no flood'.")
if data.looks_like_test_data:
    st.warning("**Test data.** Respondent IDs in this file are `Test User 1–7` (one per location) and "
               "every comment is empty, so these are placeholder records, not field responses. "
               "Every statistic below is based on n = 1 per location.", icon="⚠️")
if not data.issues.empty:
    st.warning(f"{len(data.issues)} rows failed validation and are excluded — see the **Data** tab.")

c1, c2, c3, c4 = st.columns(4)
n_resp = clean["respondent"].nunique()
n_year = int(stats["n_selected_year"].sum())
c1.metric("Selected year", year)
c2.metric(f"Respondents who ranked {year}", f"{n_year} of {n_resp}")
c3.metric(f"Locations ranking {year}", f"{int((stats['n_selected_year'] > 0).sum())} of {len(gaz)}")
ranked = clean[clean["year"] == year]["rank"]
c4.metric(f"Overall {metric.lower()} for {year}",
          "—" if ranked.empty else f"{(ranked.mean() if metric_col == 'mean_rank' else ranked.median()):.2f}")

tab_map, tab_loc, tab_data = st.tabs(["Map & years", "Location detail", "Data"])

# --------------------------------------------------------------------------- map
with tab_map:
    col_map, col_side = st.columns([3, 2], gap="large")
    with col_map:
        m = folium.Map(location=[25.25, 89.6], zoom_start=8, tiles="OpenStreetMap", control_scale=True)
        m.fit_bounds([[gaz.lat.min() - 0.25, gaz.lon.min() - 0.6], [gaz.lat.max() + 0.25, gaz.lon.max() + 0.6]])

        bounds = load_boundaries() if show_bounds else None
        if bounds:
            by_loc = stats.set_index("location")

            def style(feat):
                r = by_loc.loc[feat["properties"]["survey_name"]]
                fill = rank_color(r[metric_col]) if r["n_selected_year"] > 0 else NO_DATA
                return {"fillColor": fill, "color": fill, "weight": 1.5,
                        "fillOpacity": 0.25 if r["n_selected_year"] > 0 else 0.06,
                        "dashArray": None if r["n_selected_year"] > 0 else "4 4"}

            folium.GeoJson(bounds, name="Upazila boundaries (BBS/OCHA COD-AB)", style_function=style,
                           tooltip=folium.GeoJsonTooltip(["upazila", "district"], aliases=["Upazila", "District"])
                           ).add_to(m)

        for _, r in stats.iterrows():
            loc = r["location"]
            dist = " · ".join(f"r{k}: {r[f'rank_{k}']}" for k in P.RANKS)
            small = ("<br><b style='color:#b45309'>Small sample (n &lt; 5): interpret with caution</b>"
                     if r["small_sample"] else "")
            if r["status"] == "no survey data":
                body = "No survey responses for this location."
            elif r["status"] == "not ranked this year":
                body = (f"0 of {r['n_respondents']} respondents put {year} in their top five.<br>"
                        "This means <i>unranked</i>, not 'no flood'.")
            else:
                body = (f"<b>{r['n_selected_year']} of {r['n_respondents']}</b> respondents ranked {year}<br>"
                        f"Mean rank {r['mean_rank']:.2f} · median {r['median_rank']:.1f} (1 = worst)<br>"
                        f"Rank counts: {dist}")
            popup = (f"<div style='font-family:sans-serif;font-size:13px;min-width:230px'>"
                     f"<b style='font-size:14px'>{html.escape(loc)}</b><br>"
                     f"<span style='color:{INK_2}'>{html.escape(r['upazila'])} Upazila, "
                     f"{html.escape(r['district'])} District</span><br><br>{body}{small}</div>")
            tip = (f"{loc} — {r['n_selected_year']}/{r['n_respondents']} ranked {year}"
                   + (f", {metric.lower()} {r[metric_col]:.1f}" if r["n_selected_year"] else ""))
            selected = loc == location
            if r["status"] == "ranked":
                kw = dict(color="#ffffff", weight=2, fill_color=rank_color(r[metric_col]), fill_opacity=0.95,
                          radius=9 + 5 * (r["share_selected"] or 0))
            elif r["status"] == "not ranked this year":
                kw = dict(color=NO_DATA, weight=2, fill_color="#ffffff", fill_opacity=0.9, radius=8, dash_array="3 3")
            else:  # no survey data at all
                kw = dict(color=NO_DATA, weight=2, fill=False, radius=8)
            if selected:
                kw.update(color="#d03b3b", weight=4)
            folium.CircleMarker([r["lat"], r["lon"]], popup=folium.Popup(popup, max_width=320), tooltip=tip,
                                **kw).add_to(m)
            folium.map.Marker([r["lat"], r["lon"]], icon=folium.DivIcon(
                icon_size=(150, 20), icon_anchor=(-12, 8),
                html=f"<div style='font:600 12px sans-serif;color:{INK};text-shadow:0 0 3px #fff,0 0 3px #fff'>"
                     f"{html.escape(loc)}</div>")).add_to(m)
        folium.LayerControl(collapsed=True).add_to(m)

        out = st_folium(m, height=560, use_container_width=True, key=f"map-{year}-{metric_col}-{show_bounds}",
                        returned_objects=["last_object_clicked_tooltip"])
        clicked = (out or {}).get("last_object_clicked_tooltip")
        if clicked:
            name = clicked.split(" — ")[0].strip()
            if name in set(gaz["survey_name"]) and name != st.session_state.get("_last_click"):
                st.session_state["_last_click"] = name
                st.session_state["_clicked_location"] = name
                st.rerun()

        legend = "".join(
            f"<span style='display:inline-flex;align-items:center;margin-right:12px'>"
            f"<span style='width:14px;height:14px;border-radius:50%;background:{RANK_COLORS[k]};"
            f"display:inline-block;margin-right:5px'></span>{RANK_LABELS[k]}</span>" for k in P.RANKS)
        st.markdown(
            f"<div style='font-size:13px;color:{INK_2};line-height:2'>"
            f"<b>{metric} for {year}</b> (darker = ranked as a worse flood year): {legend}<br>"
            f"<span style='display:inline-flex;align-items:center;margin-right:14px'><span style='width:12px;height:12px;"
            f"border-radius:50%;border:2px dashed {NO_DATA};background:#fff;display:inline-block;margin-right:5px'></span>"
            f"has responses, but nobody ranked {year}</span>"
            f"<span style='display:inline-flex;align-items:center'><span style='width:12px;height:12px;border-radius:50%;"
            f"border:2px solid {NO_DATA};display:inline-block;margin-right:5px'></span>no survey responses</span><br>"
            f"Marker size grows with the share of respondents who ranked {year}. Click a marker for details.</div>",
            unsafe_allow_html=True)

    with col_side:
        st.subheader(f"{year} by location")
        tbl = stats[["location", "n_selected_year", "n_respondents", "mean_rank", "median_rank", "status"]].copy()
        tbl["respondents"] = tbl["n_selected_year"].astype(str) + " / " + tbl["n_respondents"].astype(str)
        tbl["sample"] = stats["small_sample"].map({True: "small (n<5)", False: ""})
        st.dataframe(tbl[["location", "respondents", "mean_rank", "median_rank", "status", "sample"]],
                     hide_index=True, width="stretch",
                     column_config={"mean_rank": st.column_config.NumberColumn("mean rank", format="%.2f"),
                                    "median_rank": st.column_config.NumberColumn("median rank", format="%.1f"),
                                    "respondents": st.column_config.TextColumn(f"ranked {year} / total")})
        st.caption("Mean/median are over respondents who placed this year in their top five only.")

    # ---- year frequency chart (same `view` records as the location filter)
    scope = "all locations" if location == ALL else location
    st.subheader(f"How often each year appears in the top five — {scope}")
    freq = P.year_frequency(view)
    fig = go.Figure()
    for k in P.RANKS:
        fig.add_bar(x=freq["year"], y=freq[f"rank_{k}"], name=f"Rank {RANK_LABELS[k]}",
                    marker=dict(color=RANK_COLORS[k], line=dict(color="#fcfcfb", width=1)),
                    hovertemplate="%{x}: %{y} respondent(s) ranked it " + str(k) + "<extra></extra>")
    fig.add_vrect(x0=year - 0.5, x1=year + 0.5, fillcolor="#d03b3b", opacity=0.08, line_width=0,
                  annotation_text=f"selected {year}", annotation_position="top left",
                  annotation_font=dict(size=11, color=INK_2))
    fig.update_layout(barmode="stack", height=360, margin=dict(l=10, r=10, t=30, b=10),
                      plot_bgcolor="#fcfcfb", paper_bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2),
                      legend=dict(orientation="h", y=1.12, x=0, title_text="Rank "),
                      xaxis=dict(dtick=1, tickangle=-45, title=None, showgrid=False),
                      yaxis=dict(title="respondents ranking this year", gridcolor=GRID, rangemode="tozero",
                                 dtick=1 if freq["mentions"].max() <= 10 else None))
    st.plotly_chart(fig)
    st.caption(f"Based on {view['respondent'].nunique()} respondent(s), {len(view)} ranked answers. "
               "A bar of 0 means nobody placed that year in their top five; it is not a severity of zero.")

# --------------------------------------------------------------------------- location detail
with tab_loc:
    if location == ALL:
        st.info("Choose a location in the sidebar or click a marker on the map to see its details. "
                "Showing all locations combined below.")
    loc_meta = None if location == ALL else gaz.set_index("survey_name").loc[location]
    title = "All locations" if location == ALL else f"{location} — {loc_meta['upazila']} Upazila, {loc_meta['district']}"
    st.subheader(title)
    if loc_meta is not None and loc_meta["note"]:
        st.caption(f"ℹ️ {loc_meta['note']}")

    n_loc = view["respondent"].nunique()
    a, b, c = st.columns(3)
    a.metric("Respondents", n_loc)
    b.metric("Ranked answers", len(view))
    c.metric("Distinct years mentioned", view["year"].nunique())
    if n_loc == 0:
        st.warning("No survey responses for this location in the current file.")
    else:
        if n_loc < P.SMALL_SAMPLE_N:
            st.warning(f"Small sample: {n_loc} respondent(s). Treat rankings as individual opinions, not a consensus.")
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Five most frequently reported flood years**")
            st.dataframe(P.top_years(view), hide_index=True, width="stretch",
                         column_config={"share_of_respondents": st.column_config.NumberColumn(
                             "share of respondents", format="percent"),
                             "times_ranked_worst": "times ranked #1"})
            st.caption("Ordered by number of mentions; ties broken by lower (worse) mean rank.")
        with right:
            st.markdown("**Rank distribution of the top years**")
            top = P.top_years(view)["year"].tolist()
            sub = P.year_frequency(view).set_index("year").loc[top]
            fig2 = go.Figure()
            for k in P.RANKS:
                fig2.add_bar(y=[str(y) for y in sub.index], x=sub[f"rank_{k}"], orientation="h",
                             name=f"Rank {RANK_LABELS[k]}", marker=dict(color=RANK_COLORS[k],
                                                                        line=dict(color="#fcfcfb", width=1)),
                             hovertemplate="%{y}: %{x} respondent(s) at rank " + str(k) + "<extra></extra>")
            fig2.update_layout(barmode="stack", height=280, margin=dict(l=10, r=10, t=10, b=10),
                               plot_bgcolor="#fcfcfb", paper_bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2),
                               legend=dict(orientation="h", y=-0.2), yaxis=dict(autorange="reversed", title=None),
                               xaxis=dict(title="respondents", dtick=1, gridcolor=GRID))
            st.plotly_chart(fig2)

        st.markdown("**Individual rankings**")
        wide = view.pivot_table(index=["location", "respondent"], columns="rank", values="year", aggfunc="first")
        wide.columns = [f"rank_{c}" for c in wide.columns]
        st.dataframe(wide.reset_index(), hide_index=True, width="stretch")

    st.markdown("**Written comments**")
    cm = P.comments(view)
    if cm.empty:
        st.caption("No written comments in this dataset for this selection.")
    else:
        for _, r in cm.iterrows():
            st.markdown(f"> {r['comment']}\n\n— *{r['respondent']}, {r['location']}*")

# --------------------------------------------------------------------------- data
with tab_data:
    st.subheader("Source file")
    st.code(Path(data.path).name)
    st.markdown(f"**Format detected:** {data.info['format']}")
    st.json(data.info["columns_used"])
    for n in data.info["notes"]:
        st.caption(f"• {n}")
    st.markdown("**Schema**")
    st.dataframe(data.schema, hide_index=True, width="stretch")
    st.markdown(f"**Validated records ({len(clean)})**")
    st.dataframe(clean, hide_index=True, width="stretch")
    st.markdown(f"**Rejected rows ({len(data.issues)})**")
    if data.issues.empty:
        st.caption("None — every row passed validation (year 2000–2020, rank 1–5, known location, "
                   "no duplicate rank or year per respondent).")
    else:
        st.dataframe(data.issues, hide_index=True, width="stretch")
    st.markdown("**Location gazetteer** (coordinates from BBS/OCHA COD-AB upazila polygons; see data/SOURCES.md)")
    st.dataframe(gaz, hide_index=True, width="stretch")

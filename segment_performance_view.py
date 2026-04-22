import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import altair as alt
from db import load_gtfs_segment_events
from loaders import load_gtfs_stop_tables


def _normalize_direction_id(value) -> str:
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def build_static_segments_for_route(
    gtfs_static_path: str,
    route_id: str,
    direction_id: str | None = None,
) -> pd.DataFrame:
    stops_df, stop_times_df, trips_df = load_gtfs_stop_tables(gtfs_static_path)

    if stops_df.empty or stop_times_df.empty or trips_df.empty:
        return pd.DataFrame()

    trips_df = trips_df.copy()
    trips_df["route_id"] = trips_df["route_id"].astype(str).str.strip()
    trips_df["trip_id"] = trips_df["trip_id"].astype(str).str.strip()

    if "direction_id" in trips_df.columns:
        trips_df["direction_id"] = (
            trips_df["direction_id"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
    else:
        trips_df["direction_id"] = "unknown"

    route_id = str(route_id).strip()
    trips_sel = trips_df[trips_df["route_id"] == route_id].copy()

    if direction_id is not None and direction_id != "All":
        direction_id = _normalize_direction_id(direction_id)
        trips_sel = trips_sel[trips_sel["direction_id"] == direction_id].copy()

    if trips_sel.empty:
        return pd.DataFrame()

    stop_times_df = stop_times_df.copy()
    stop_times_df["trip_id"] = stop_times_df["trip_id"].astype(str).str.strip()
    stop_times_df["stop_id"] = stop_times_df["stop_id"].astype(str).str.strip()
    stop_times_df["stop_sequence"] = pd.to_numeric(stop_times_df["stop_sequence"], errors="coerce")

    st_sel = stop_times_df.merge(
        trips_sel[["trip_id", "route_id", "direction_id"]],
        on="trip_id",
        how="inner",
    )

    if st_sel.empty:
        return pd.DataFrame()

    # representative pattern: longest trip by max stop_sequence
    trip_lengths = (
        st_sel.groupby("trip_id")["stop_sequence"]
        .max()
        .sort_values(ascending=False)
    )
    rep_trip_id = trip_lengths.index[0]

    rep = st_sel[st_sel["trip_id"] == rep_trip_id].copy()
    rep = rep.sort_values("stop_sequence")

    rep["next_stop_id"] = rep["stop_id"].shift(-1)
    rep["next_stop_sequence"] = rep["stop_sequence"].shift(-1)
    rep = rep.dropna(subset=["next_stop_id"]).copy()

    stops_small = stops_df.copy()
    stops_small["stop_id"] = stops_small["stop_id"].astype(str).str.strip()
    if "stop_name" not in stops_small.columns:
        stops_small["stop_name"] = stops_small["stop_id"]

    rep = rep.merge(
        stops_small[["stop_id", "stop_name", "stop_lat", "stop_lon"]],
        on="stop_id",
        how="left",
    ).rename(
        columns={
            "stop_name": "from_stop_name",
            "stop_lat": "from_stop_lat",
            "stop_lon": "from_stop_lon",
        }
    )

    rep = rep.merge(
        stops_small[["stop_id", "stop_name", "stop_lat", "stop_lon"]],
        left_on="next_stop_id",
        right_on="stop_id",
        how="left",
        suffixes=("", "_to"),
    ).rename(
        columns={
            "next_stop_id": "to_stop_id",
            "stop_name": "to_stop_name",
            "stop_lat": "to_stop_lat",
            "stop_lon": "to_stop_lon",
        }
    )

    rep = rep.rename(columns={"stop_id": "from_stop_id"})
    rep = rep[
        [
            "route_id",
            "direction_id",
            "from_stop_id",
            "to_stop_id",
            "from_stop_name",
            "to_stop_name",
            "from_stop_lat",
            "from_stop_lon",
            "to_stop_lat",
            "to_stop_lon",
            "stop_sequence",
            "next_stop_sequence",
        ]
    ].copy()

    rep["segment_label"] = rep.apply(
        lambda r: f"{r['from_stop_name']} → {r['to_stop_name']} ({r['from_stop_id']} → {r['to_stop_id']})",
        axis=1,
    )

    return rep.reset_index(drop=True)


def make_segment_map(segment_row: pd.Series):
    center_lat = (segment_row["from_stop_lat"] + segment_row["to_stop_lat"]) / 2
    center_lon = (segment_row["from_stop_lon"] + segment_row["to_stop_lon"]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="cartodbpositron")

    folium.CircleMarker(
        location=[segment_row["from_stop_lat"], segment_row["from_stop_lon"]],
        radius=6,
        color="#1f77b4",
        fill=True,
        fill_color="#1f77b4",
        fill_opacity=0.95,
        tooltip=f"From: {segment_row['from_stop_name']} ({segment_row['from_stop_id']})",
    ).add_to(m)

    folium.CircleMarker(
        location=[segment_row["to_stop_lat"], segment_row["to_stop_lon"]],
        radius=6,
        color="#d62728",
        fill=True,
        fill_color="#d62728",
        fill_opacity=0.95,
        tooltip=f"To: {segment_row['to_stop_name']} ({segment_row['to_stop_id']})",
    ).add_to(m)

    folium.PolyLine(
        locations=[
            [segment_row["from_stop_lat"], segment_row["from_stop_lon"]],
            [segment_row["to_stop_lat"], segment_row["to_stop_lon"]],
        ],
        color="#ff7f0e",
        weight=5,
        opacity=0.9,
    ).add_to(m)

    return m


def build_histogram_from_events(events_df: pd.DataFrame) -> pd.DataFrame:
    labels = [
        "0–1",
        "1–2",
        "2–3",
        "3–4",
        "4–5",
        "5–7",
        "7–10",
        "10–15",
        "15–20",
        "20–30",
        "30–45",
        "45–60",
        "60–100",
        "100+",
    ]

    if events_df.empty or "tt_min" not in events_df.columns:
        return pd.DataFrame({"bin_label": labels, "n_obs": [0] * len(labels)})

    df = events_df.copy()
    df["tt_min"] = pd.to_numeric(df["tt_min"], errors="coerce")
    df = df.dropna(subset=["tt_min"]).copy()

    df = df[(df["tt_min"] >= 0) & (df["tt_min"] <= 100)].copy()

    if df.empty:
        return pd.DataFrame({"bin_label": labels, "n_obs": [0] * len(labels)})

    bins = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 45, 60, 100, float("inf")]

    df["bin_label"] = pd.cut(
        df["tt_min"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
        ordered=True,
    )

    out = (
        df.groupby("bin_label", observed=False)
        .size()
        .reindex(labels, fill_value=0)
        .reset_index(name="n_obs")
    )

    out["bin_label"] = pd.Categorical(out["bin_label"], categories=labels, ordered=True)
    return out


def build_trend_from_events(events_df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    if events_df.empty or "traversal_time" not in events_df.columns or "tt_min" not in events_df.columns:
        
        return pd.DataFrame(columns=["traversal_time", "avg_tt_min", "median_tt_min", "p95_tt_min", "n_trips"])

    df = events_df.copy()
    df["tt_min"] = pd.to_numeric(df["tt_min"], errors="coerce")
    df = df.dropna(subset=["traversal_time", "tt_min"]).copy()
    df = df[(df["tt_min"] >= 0) & (df["tt_min"] <= 100)].copy() 
     

    if df.empty:
        return pd.DataFrame(columns=["traversal_time", "avg_tt_min", "median_tt_min", "p95_tt_min", "n_trips"])

    def _p95(x):
        if len(x) == 0:
            return None
        return x.quantile(0.95)

    grouped = (
        df.set_index("traversal_time")
        .groupby(pd.Grouper(freq=freq))
        .agg(
            avg_tt_min=("tt_min", "mean"),
            median_tt_min=("tt_min", "median"),
            min_tt_min=("tt_min", "min"),
            max_tt_min=("tt_min", "max"),
            n_obs=("tt_min", "size"),
            n_trips=("trip_id", pd.Series.nunique),
        )
        .reset_index()
    )

    p95_df = (
        df.set_index("traversal_time")
        .groupby(pd.Grouper(freq=freq))["tt_min"]
        .apply(_p95)
        .reset_index(name="p95_tt_min")
    )

    out = grouped.merge(p95_df, on="traversal_time", how="left")
    out = out.sort_values("traversal_time").reset_index(drop=True)

    return out


def render_segment_performance_view(gtfs_static_path: str):
    st.subheader("Segment performance")

    route_id = st.text_input("Route ID", value="", key="seg_route_id").strip()
    direction_id = st.text_input("Direction ID (optional)", value="", key="seg_direction_id").strip()

    if not route_id:
        st.info("Insert route_id first.")
        return

    direction_value = None if direction_id == "" else direction_id

    segments_df = build_static_segments_for_route(
        gtfs_static_path=gtfs_static_path,
        route_id=route_id,
        direction_id=direction_value,
    )

    if segments_df.empty:
        st.warning("No static segments found for this route/direction.")
        return

    segment_label = st.selectbox(
        "Choose segment",
        segments_df["segment_label"].tolist(),
        index=0,
        key="seg_segment_label",
    )

    selected_segment = segments_df.loc[segments_df["segment_label"] == segment_label].iloc[0]
    from_stop_id = str(selected_segment["from_stop_id"]).strip()
    to_stop_id = str(selected_segment["to_stop_id"]).strip()

    period_label = st.selectbox(
        "Analysis period",
        ["Last 24 hours", "Last 7 days"],
        index=0,
        key="seg_period",
    )
    hours_back = 24 if period_label == "Last 24 hours" else 24 * 7

    trend_freq_label = st.selectbox(
        "Trend aggregation",
        ["15 min", "30 min", "1 hour"],
        index=0,
        key="seg_trend_freq",
    )
    trend_freq = {"15 min": "15min", "30 min": "30min", "1 hour": "1h"}[trend_freq_label]

    events_df = load_gtfs_segment_events(
        route_id=route_id,
        direction_id=direction_value,
        from_stop_id=from_stop_id,
        to_stop_id=to_stop_id,
        hours_back=hours_back,
    )

    segment_map = make_segment_map(selected_segment)

    st.markdown(
        f"**Segment:** {selected_segment['from_stop_name']} → {selected_segment['to_stop_name']}  "
        f"(`{from_stop_id}` → `{to_stop_id}`)"
    )
    st_folium(segment_map, width=None, height=420)

    if events_df.empty:
        st.warning("No observed segment traversals found for this selection and period.")
        return

    trend_df = build_trend_from_events(events_df, freq=trend_freq)
    hist_df = build_histogram_from_events(events_df)

    latest_tt_mean = round(float(events_df["tt_min"].mean()), 2) if "tt_min" in events_df.columns else None
    latest_tt_median = round(float(events_df["tt_min"].median()), 2) if "tt_min" in events_df.columns else None
    latest_tt_p95 = round(float(events_df["tt_min"].quantile(0.95)), 2) if "tt_min" in events_df.columns else None
    latest_n_trips = events_df["trip_id"].nunique() if "trip_id" in events_df.columns else None
    latest_n_obs = len(events_df)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Route", route_id)
    c2.metric("Direction", direction_value if direction_value is not None else "All")
    c3.metric("Distinct trips", latest_n_trips if latest_n_trips is not None else "—")
    c4.metric("Observed traversals", latest_n_obs)
    c5.metric("P95 TT (min)", latest_tt_p95 if latest_tt_p95 is not None else "—")

    c6, c7 = st.columns(2)
    c6.metric("Mean TT (min)", latest_tt_mean if latest_tt_mean is not None else "—")
    c7.metric("Median TT (min)", latest_tt_median if latest_tt_median is not None else "—")

    st.subheader(f"Trend over {period_label.lower()}")
    if trend_df.empty:
        st.info("No trend data available.")
    else:
        chart_cols = [c for c in ["avg_tt_min", "median_tt_min", "p95_tt_min", "n_trips"] if c in trend_df.columns]
        chart_df = trend_df.set_index("traversal_time")[chart_cols]
        st.line_chart(chart_df, height=320)


    st.subheader(f"Histogram over {period_label.lower()}")
    if hist_df.empty:
        st.info("No histogram data available.")
    else:
        hist_plot_df = hist_df.copy()
        bin_order = [
        "0–1",
        "1–2",
        "2–3",
        "3–4",
        "4–5",
        "5–7",
        "7–10",
        "10–15",
        "15–20",
        "20–30",
        "30–45",
        "45–60",
        "60–100",
        "100+",
    ]

    chart = (
        alt.Chart(hist_plot_df)
        .mark_bar()
        .encode(
            x=alt.X("bin_label:N", sort=bin_order, title="Travel time bin (min)"),
            y=alt.Y("n_obs:Q", title="Observed traversals"),
            tooltip=["bin_label", "n_obs"],
        )
        .properties(height=320)
    )

    st.altair_chart(chart, use_container_width=True)

    with st.expander("Trend table", expanded=False):
        st.dataframe(trend_df, use_container_width=True)

    with st.expander("Histogram table", expanded=False):
        st.dataframe(hist_df, use_container_width=True)

    with st.expander("Raw segment traversals", expanded=False):
        st.dataframe(events_df, use_container_width=True)

    with st.expander("Static segment info", expanded=False):
        st.dataframe(pd.DataFrame([selected_segment]), use_container_width=True)
import json
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from google.transit import gtfs_realtime_pb2
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh

from config import GTFS_HISTORY_PATH, MAP_CENTER, MAP_ZOOM, TRIP_UPDATES_URL, VEHICLE_URL
from loaders import (
    fetch_gtfs_rt_vehicle_positions,
    join_vehicle_positions_with_routes,
    load_gtfs_static,
)
from map_utils import add_focus_css


def add_small_metric_css():
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background-color: #eef2f7 !important;
            border: 1px solid #c7d0db !important;
            border-left: 4px solid #5b6573 !important;
            padding: 10px 12px !important;
            border-radius: 10px !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.06) !important;
        }
        div[data-testid="stMetric"] * {
            color: #102a43 !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.80rem !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.20rem !important;
            line-height: 1.15 !important;
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def decode_vehicle_status(x):
    try:
        x = int(x)
    except Exception:
        return "unknown"

    mapping = {
        0: "incoming at",
        1: "stopped at",
        2: "in transit to",
    }
    return mapping.get(x, f"unknown ({x})")


def classify_delay_status(delay_sec):
    if pd.isna(delay_sec):
        return "unknown"
    if delay_sec > 60:
        return "delayed"
    if delay_sec < -60:
        return "early"
    return "on time"


def delay_color(delay_status: str) -> str:
    delay_status = str(delay_status).strip().lower()

    if delay_status == "delayed":
        return "#de2d26"
    if delay_status == "on time":
        return "#31a354"
    if delay_status == "early":
        return "#3182bd"
    return "#666666"


@st.cache_data(ttl=15, show_spinner=False)
def fetch_gtfs_rt_trip_updates(trip_updates_url: str) -> pd.DataFrame:
    resp = requests.get(trip_updates_url, timeout=20)
    resp.raise_for_status()

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(resp.content)

    rows = []
    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue

        tu = entity.trip_update
        trip = tu.trip if tu.HasField("trip") else None
        vehicle = tu.vehicle if tu.HasField("vehicle") else None

        trip_id = trip.trip_id if trip and trip.trip_id else None
        route_id = trip.route_id if trip and trip.route_id else None
        vehicle_id = vehicle.id if vehicle and vehicle.id else None

        trip_delay = None
        if tu.HasField("delay"):
            trip_delay = int(tu.delay)

        best_stop_id = None
        best_delay = trip_delay

        if len(tu.stop_time_update) > 0:
            stu = tu.stop_time_update[0]
            if stu.stop_id:
                best_stop_id = stu.stop_id

            if stu.HasField("arrival") and stu.arrival.HasField("delay"):
                best_delay = int(stu.arrival.delay)
            elif stu.HasField("departure") and stu.departure.HasField("delay"):
                best_delay = int(stu.departure.delay)

        rows.append(
            {
                "trip_id": trip_id,
                "route_id": route_id,
                "vehicle_id": vehicle_id,
                "update_stop_id": best_stop_id,
                "delay_sec": best_delay,
                "delay_min": None if best_delay is None else round(best_delay / 60.0, 1),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if "trip_id" in df.columns:
        df["trip_id"] = df["trip_id"].astype(str)

    df["delay_status"] = df["delay_sec"].apply(classify_delay_status)
    return df


def enrich_vehicle_positions_with_delay(vehicle_df: pd.DataFrame, trip_updates_df: pd.DataFrame) -> pd.DataFrame:
    if vehicle_df.empty:
        return vehicle_df.copy()

    out = vehicle_df.copy()
    out["trip_id"] = out["trip_id"].astype(str)

    if trip_updates_df.empty:
        out["delay_sec"] = pd.NA
        out["delay_min"] = pd.NA
        out["delay_status"] = "unknown"
        return out

    trip_updates_df = trip_updates_df.copy()
    trip_updates_df["trip_id"] = trip_updates_df["trip_id"].astype(str)

    keep_cols = ["trip_id", "delay_sec", "delay_min", "delay_status", "update_stop_id"]
    merged = out.merge(
        trip_updates_df[keep_cols].drop_duplicates("trip_id"),
        on="trip_id",
        how="left",
    )

    merged["delay_status"] = merged["delay_status"].fillna("unknown")
    return merged


def build_overall_gtfs_stats(vehicle_df: pd.DataFrame) -> pd.DataFrame:
    if vehicle_df.empty:
        return pd.DataFrame(
            [
                {
                    "total_buses": 0,
                    "unique_routes": 0,
                    "delayed": 0,
                    "on_time": 0,
                    "early": 0,
                    "unknown_delay": 0,
                    "pct_delayed": 0.0,
                    "pct_on_time": 0.0,
                    "pct_early": 0.0,
                    "pct_unknown_delay": 0.0,
                    "avg_delay_min": None,
                    "p95_delay_min": None,
                    "min_delay_min": None,
                    "max_delay_min": None,
                }
            ]
        )

    df = vehicle_df.copy()

    if "route_id" not in df.columns:
        df["route_id"] = "unknown"
    if "direction_id" not in df.columns:
        df["direction_id"] = "unknown"

    df["route_id"] = df["route_id"].fillna("unknown").astype(str)
    df["direction_id"] = df["direction_id"].fillna("unknown").astype(str)
    df["delay_status"] = df["delay_status"].fillna("unknown")
    df["delay_min"] = pd.to_numeric(df["delay_min"], errors="coerce")

    total_buses = df["vehicle_id"].nunique() if "vehicle_id" in df.columns else len(df)
    unique_routes = df[["route_id", "direction_id"]].drop_duplicates().shape[0]

    delayed = int((df["delay_status"] == "delayed").sum())
    on_time = int((df["delay_status"] == "on time").sum())
    early = int((df["delay_status"] == "early").sum())
    unknown_delay = int((df["delay_status"] == "unknown").sum())

    denom = len(df) if len(df) > 0 else 1
    positive_delays = df.loc[df["delay_min"] > 0, "delay_min"].dropna()

    return pd.DataFrame(
        [
            {
                "total_buses": total_buses,
                "unique_routes": unique_routes,
                "delayed": delayed,
                "on_time": on_time,
                "early": early,
                "unknown_delay": unknown_delay,
                "pct_delayed": round(100.0 * delayed / denom, 1),
                "pct_on_time": round(100.0 * on_time / denom, 1),
                "pct_early": round(100.0 * early / denom, 1),
                "pct_unknown_delay": round(100.0 * unknown_delay / denom, 1),
                "avg_delay_min": round(positive_delays.mean(), 1) if not positive_delays.empty else None,
                "p95_delay_min": round(float(positive_delays.quantile(0.95)), 1) if not positive_delays.empty else None,
                "min_delay_min": round(positive_delays.min(), 1) if not positive_delays.empty else None,
                "max_delay_min": round(positive_delays.max(), 1) if not positive_delays.empty else None,
            }
        ]
    )


def build_route_direction_stats(vehicle_df: pd.DataFrame) -> pd.DataFrame:
    if vehicle_df.empty:
        return pd.DataFrame(
            columns=[
                "route_id",
                "route_label",
                "direction_id",
                "n_buses",
                "delayed",
                "on_time",
                "early",
                "unknown_delay",
                "avg_delay_min",
                "p95_delay_min",
                "min_delay_min",
                "max_delay_min",
                "incoming_at",
                "stopped_at",
                "in_transit_to",
            ]
        )

    df = vehicle_df.copy()

    if "route_label" not in df.columns:
        df["route_label"] = df["route_id"]

    if "direction_id" not in df.columns:
        df["direction_id"] = "unknown"

    df["route_id"] = df["route_id"].fillna("unknown").astype(str)
    df["route_label"] = df["route_label"].fillna(df["route_id"]).astype(str)
    df["direction_id"] = df["direction_id"].fillna("unknown").astype(str)
    df["delay_status"] = df["delay_status"].fillna("unknown")
    df["status_text"] = df["status_text"].fillna("unknown")
    df["delay_min"] = pd.to_numeric(df["delay_min"], errors="coerce")

    vehicle_id_col = "vehicle_id" if "vehicle_id" in df.columns else "trip_id"
    positive_delay = df["delay_min"].where(df["delay_min"] > 0)

    grouped = (
        df.assign(_positive_delay=positive_delay)
        .groupby(["route_id", "route_label", "direction_id"], dropna=False)
        .agg(
            n_buses=(vehicle_id_col, "nunique"),
            delayed=("delay_status", lambda s: int((s == "delayed").sum())),
            on_time=("delay_status", lambda s: int((s == "on time").sum())),
            early=("delay_status", lambda s: int((s == "early").sum())),
            unknown_delay=("delay_status", lambda s: int((s == "unknown").sum())),
            avg_delay_min=("_positive_delay", "mean"),
            p95_delay_min=("_positive_delay", lambda s: s.dropna().quantile(0.95) if not s.dropna().empty else np.nan),
            min_delay_min=("_positive_delay", "min"),
            max_delay_min=("_positive_delay", "max"),
            incoming_at=("status_text", lambda s: int((s == "incoming at").sum())),
            stopped_at=("status_text", lambda s: int((s == "stopped at").sum())),
            in_transit_to=("status_text", lambda s: int((s == "in transit to").sum())),
        )
        .reset_index()
        .sort_values(
            ["delayed", "n_buses", "route_id", "direction_id"],
            ascending=[False, False, True, True],
        )
        .reset_index(drop=True)
    )

    for col in ["avg_delay_min", "p95_delay_min", "min_delay_min", "max_delay_min"]:
        grouped[col] = grouped[col].round(1)

    return grouped


def update_hourly_delay_store(vehicle_df: pd.DataFrame, stats_path: Path = GTFS_HISTORY_PATH) -> None:
    """
    Persist lightweight snapshot aggregates for later 24h profiling.
    History is stored from the full unfiltered live feed.
    Delay magnitude metrics exclude negative values.
    """
    if vehicle_df.empty or "timestamp_dt" not in vehicle_df.columns:
        return

    df = vehicle_df.copy()
    df["timestamp_dt"] = pd.to_datetime(df["timestamp_dt"], errors="coerce")
    df = df.dropna(subset=["timestamp_dt"]).copy()
    if df.empty:
        return

    ts = df["timestamp_dt"].max()
    if pd.isna(ts):
        return

    df["delay_status"] = df["delay_status"].fillna("unknown")
    df["delay_min"] = pd.to_numeric(df["delay_min"], errors="coerce")

    n_buses = int(df["vehicle_id"].nunique()) if "vehicle_id" in df.columns else int(len(df))
    delayed = int((df["delay_status"] == "delayed").sum())
    pct_delayed = round(100.0 * delayed / n_buses, 2) if n_buses > 0 else 0.0

    positive_delays = df.loc[df["delay_min"] > 0, "delay_min"].dropna()

    avg_delay_min = round(positive_delays.mean(), 2) if not positive_delays.empty else np.nan
    p95_delay_min = round(float(positive_delays.quantile(0.95)), 2) if not positive_delays.empty else np.nan
    min_delay_min = round(positive_delays.min(), 2) if not positive_delays.empty else np.nan
    max_delay_min = round(positive_delays.max(), 2) if not positive_delays.empty else np.nan

    snapshot_time = ts.floor("min")
    row = pd.DataFrame(
        [
            {
                "snapshot_time": snapshot_time,
                "date": snapshot_time.date().isoformat(),
                "hour": int(snapshot_time.hour),
                "n_buses": n_buses,
                "delayed": delayed,
                "pct_delayed": pct_delayed,
                "avg_delay_min": avg_delay_min,
                "p95_delay_min": p95_delay_min,
                "min_delay_min": min_delay_min,
                "max_delay_min": max_delay_min,
            }
        ]
    )

    stats_path.parent.mkdir(parents=True, exist_ok=True)

    if stats_path.exists():
        try:
            old = pd.read_parquet(stats_path)
            combined = pd.concat([old, row], ignore_index=True)
        except Exception:
            combined = row.copy()
    else:
        combined = row.copy()

    combined["snapshot_time"] = pd.to_datetime(combined["snapshot_time"], errors="coerce")
    combined = combined.drop_duplicates(subset=["snapshot_time"], keep="last")
    combined = combined.sort_values("snapshot_time").reset_index(drop=True)

    combined.to_parquet(stats_path, index=False)


def load_hourly_delay_store(stats_path: Path = GTFS_HISTORY_PATH) -> pd.DataFrame:
    empty = pd.DataFrame(
        columns=[
            "snapshot_time",
            "date",
            "hour",
            "n_buses",
            "delayed",
            "pct_delayed",
            "avg_delay_min",
            "p95_delay_min",
            "min_delay_min",
            "max_delay_min",
        ]
    )

    if not stats_path.exists():
        return empty

    try:
        df = pd.read_parquet(stats_path)
    except Exception:
        return empty

    if "snapshot_time" in df.columns:
        df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], errors="coerce")

    return df


def build_last_24h_delay_profile(stats_store_df: pd.DataFrame) -> pd.DataFrame:
    if stats_store_df.empty:
        now_hour = pd.Timestamp.now().floor("h")
        full_hours = pd.date_range(end=now_hour, periods=24, freq="h")
        return pd.DataFrame(
            {
                "hour_ts": full_hours,
                "avg_pct_delayed": [0.0] * 24,
                "avg_delay_min": [0.0] * 24,
                "p95_delay_min": [0.0] * 24,
                "n_snapshots": [0] * 24,
            }
        )

    df = stats_store_df.copy()
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], errors="coerce")
    df["pct_delayed"] = pd.to_numeric(df["pct_delayed"], errors="coerce")
    df["avg_delay_min"] = pd.to_numeric(df["avg_delay_min"], errors="coerce")
    df["p95_delay_min"] = pd.to_numeric(df["p95_delay_min"], errors="coerce")
    df = df.dropna(subset=["snapshot_time"]).copy()

    if df.empty:
        now_hour = pd.Timestamp.now().floor("h")
        full_hours = pd.date_range(end=now_hour, periods=24, freq="h")
        return pd.DataFrame(
            {
                "hour_ts": full_hours,
                "avg_pct_delayed": [0.0] * 24,
                "avg_delay_min": [0.0] * 24,
                "p95_delay_min": [0.0] * 24,
                "n_snapshots": [0] * 24,
            }
        )

    max_ts = df["snapshot_time"].max().floor("h")
    min_ts = max_ts - pd.Timedelta(hours=23)

    df = df[(df["snapshot_time"] >= min_ts) & (df["snapshot_time"] < max_ts + pd.Timedelta(hours=1))].copy()

    if df.empty:
        full_hours = pd.date_range(start=min_ts, end=max_ts, freq="h")
        return pd.DataFrame(
            {
                "hour_ts": full_hours,
                "avg_pct_delayed": [0.0] * len(full_hours),
                "avg_delay_min": [0.0] * len(full_hours),
                "p95_delay_min": [0.0] * len(full_hours),
                "n_snapshots": [0] * len(full_hours),
            }
        )

    df["hour_ts"] = df["snapshot_time"].dt.floor("h")

    grouped = (
        df.groupby("hour_ts", as_index=False)
        .agg(
            avg_pct_delayed=("pct_delayed", "mean"),
            avg_delay_min=("avg_delay_min", "mean"),
            p95_delay_min=("p95_delay_min", "mean"),
            n_snapshots=("snapshot_time", "nunique"),
        )
    )

    full_hours = pd.DataFrame({"hour_ts": pd.date_range(start=min_ts, end=max_ts, freq="h")})
    grouped = full_hours.merge(grouped, on="hour_ts", how="left").fillna(0)

    grouped["avg_pct_delayed"] = grouped["avg_pct_delayed"].round(1)
    grouped["avg_delay_min"] = grouped["avg_delay_min"].round(1)
    grouped["p95_delay_min"] = grouped["p95_delay_min"].round(1)
    grouped["n_snapshots"] = grouped["n_snapshots"].astype(int)

    return grouped


def make_live_buses_map(
    vehicle_df: pd.DataFrame,
    stops_gdf=None,
    show_stops: bool = False,
):
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles="cartodbpositron")
    add_focus_css(m)

    if show_stops and stops_gdf is not None and not stops_gdf.empty:
        stops_sample = stops_gdf.copy()
        if len(stops_sample) > 2000:
            stops_sample = stops_sample.sample(2000, random_state=42)

        for _, row in stops_sample.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=1.5,
                color="#666666",
                weight=1,
                fill=True,
                fill_opacity=0.35,
                tooltip=str(row.get("stop_name", row.get("stop_id", ""))),
            ).add_to(m)

    if not vehicle_df.empty:
        for _, row in vehicle_df.iterrows():
            route_label = row.get("route_label", row.get("route_id", ""))
            vehicle_status = row.get("status_text", "unknown")
            delay_status = row.get("delay_status", "unknown")
            delay_min = row.get("delay_min", None)
            direction_id = row.get("direction_id", "unknown")

            delay_text = "unknown" if pd.isna(delay_min) else f"{delay_min} min"
            marker_color = delay_color(delay_status)

            tooltip = folium.Tooltip(
                (
                    f"Route: {route_label}<br>"
                    f"Route ID: {row.get('route_id', '')}<br>"
                    f"Direction ID: {direction_id}<br>"
                    f"Trip: {row.get('trip_id', '')}<br>"
                    f"Vehicle: {row.get('vehicle_label', row.get('vehicle_id', ''))}<br>"
                    f"Vehicle status: {vehicle_status}<br>"
                    f"Delay status: {delay_status}<br>"
                    f"Delay: {delay_text}<br>"
                    f"Stop ID: {row.get('stop_id', '')}<br>"
                    f"Time: {row.get('timestamp_dt', '')}"
                )
            )

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3,
                color="#111111",
                weight=1,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.9,
                tooltip=tooltip,
            ).add_to(m)

    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background: rgba(255,255,255,0.96);
        color: #111111;
        border: 1px solid #999999;
        border-radius: 6px;
        padding: 10px 12px;
        font-size: 12px;
        line-height: 1.4;
        box-shadow: 0 1px 6px rgba(0,0,0,0.2);
        ">
        <div style="font-weight: 700; color: #111111; margin-bottom: 6px;">Delay status</div>
        <div style="color: #111111;"><span style="color:#de2d26; font-weight:700;">●</span> delayed</div>
        <div style="color: #111111;"><span style="color:#31a354; font-weight:700;">●</span> on time</div>
        <div style="color: #111111;"><span style="color:#3182bd; font-weight:700;">●</span> early</div>
        <div style="color: #111111;"><span style="color:#666666; font-weight:700;">●</span> unknown</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def render_gtfs_rt_view(zones, zone_field, choices, gtfs_static_path):
    with st.sidebar:
        show_stops = st.toggle("Show GTFS stops", value=False, key="gtfs_show_stops")
        auto_refresh = st.slider("Refresh every N sec", 5, 60, 15, 5, key="gtfs_refresh_sec")
        only_route = st.text_input("Filter by route_id (optional)", value="", key="gtfs_route_filter")
        only_delay_status = st.multiselect(
            "Filter by delay status",
            options=["delayed", "on time", "early", "unknown"],
            default=[],
            key="gtfs_delay_status_filter",
        )

    st_autorefresh(interval=auto_refresh * 1000, key="gtfs_live_refresh")

    routes_df, stops_gdf = load_gtfs_static(gtfs_static_path)
    vehicle_df = fetch_gtfs_rt_vehicle_positions(VEHICLE_URL)
    vehicle_df = join_vehicle_positions_with_routes(vehicle_df, routes_df)

    if not vehicle_df.empty:
        vehicle_df["status_text"] = vehicle_df["current_status"].apply(decode_vehicle_status)
    else:
        vehicle_df["status_text"] = pd.Series(dtype="object")

    trip_updates_df = fetch_gtfs_rt_trip_updates(TRIP_UPDATES_URL)
    vehicle_df = enrich_vehicle_positions_with_delay(vehicle_df, trip_updates_df)

    # Store history from the full live dataset before applying user filters.
    vehicle_df_all = vehicle_df.copy()
    update_hourly_delay_store(vehicle_df_all, GTFS_HISTORY_PATH)
    hourly_store_df = load_hourly_delay_store(GTFS_HISTORY_PATH)
    hourly_delay_df = build_last_24h_delay_profile(hourly_store_df)

    # Filters affect display only, not stored history.
    if only_route.strip():
        vehicle_df = vehicle_df[vehicle_df["route_id"].astype(str) == only_route.strip()].copy()

    if only_delay_status:
        vehicle_df = vehicle_df[vehicle_df["delay_status"].isin(only_delay_status)].copy()

    overall_stats_df = build_overall_gtfs_stats(vehicle_df)
    stats_df = build_route_direction_stats(vehicle_df)

    add_small_metric_css()
    c1, c2, c3 = st.columns(3)
    c1.metric("Live vehicles", len(vehicle_df))
    c2.metric("Refresh (sec)", auto_refresh)
    c3.metric("Active route filter", only_route.strip() if only_route.strip() else "None")

    m = make_live_buses_map(
        vehicle_df=vehicle_df,
        stops_gdf=stops_gdf,
        show_stops=show_stops,
    )
    st_folium(m, width=None, height=760)

    st.subheader("Delayed buses in the last 24 hours")
    st.line_chart(
        hourly_delay_df.set_index("hour_ts")[["avg_pct_delayed"]],
        height=260,
    )

    st.subheader("Positive delay magnitude in the last 24 hours")
    st.line_chart(
        hourly_delay_df.set_index("hour_ts")[["avg_delay_min", "p95_delay_min"]],
        height=260,
    )

    st.caption(
        "Average delay and P95 delay are computed only on positive delays."
    )
    with st.expander("Overall GTFS-RT summary", expanded=True):
        st.dataframe(overall_stats_df, use_container_width=True)

    with st.expander("Route and direction statistics", expanded=True):
        if stats_df.empty:
            st.write("No statistics available.")
        else:
            st.dataframe(stats_df, use_container_width=True)

    with st.expander("Last 24 hours delay table", expanded=False):
        st.dataframe(hourly_delay_df, use_container_width=True)

    with st.expander("Live vehicle table"):
        if vehicle_df.empty:
            st.write("No live vehicles found.")
        else:
            show_cols = [
                c
                for c in [
                    "route_label",
                    "route_id",
                    "direction_id",
                    "trip_id",
                    "vehicle_id",
                    "vehicle_label",
                    "status_text",
                    "delay_status",
                    "delay_sec",
                    "delay_min",
                    "stop_id",
                    "update_stop_id",
                    "latitude",
                    "longitude",
                    "speed_m_s",
                    "bearing",
                    "timestamp_dt",
                ]
                if c in vehicle_df.columns
            ]
            st.dataframe(
                vehicle_df[show_cols].sort_values(
                    ["route_id", "direction_id", "delay_status"],
                    na_position="last",
                ),
                use_container_width=True,
            )
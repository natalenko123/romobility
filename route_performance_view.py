from __future__ import annotations

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from config import TRIP_UPDATES_URL, VEHICLE_URL
from db import load_gtfs_route_last_24h
from gtfs_rt_view import (
    build_overall_gtfs_stats,
    decode_vehicle_status,
    enrich_vehicle_positions_with_delay,
    fetch_gtfs_rt_trip_updates,
)
from loaders import (
    fetch_gtfs_rt_vehicle_positions,
    join_vehicle_positions_with_routes,
    load_gtfs_route_map_data,
    load_gtfs_static,
)

def _clean_route_delay_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    for col in ["avg_delay_min", "p95_delay_min", "min_delay_min", "max_delay_min"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # keep plausible but broad range for display
    if "avg_delay_min" in out.columns:
        out.loc[out["avg_delay_min"].abs() > 100, "avg_delay_min"] = pd.NA
    if "p95_delay_min" in out.columns:
        out.loc[out["p95_delay_min"].abs() > 100, "p95_delay_min"] = pd.NA
    if "min_delay_min" in out.columns:
        out.loc[out["min_delay_min"].abs() > 100, "min_delay_min"] = pd.NA
    if "max_delay_min" in out.columns:
        out.loc[out["max_delay_min"].abs() > 100, "max_delay_min"] = pd.NA

    return out

def _normalize_direction_id(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"none", "nan"}:
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _build_route_label(row: pd.Series) -> str:
    route_id = str(row.get("route_id", "")).strip()
    short_name = str(row.get("route_short_name", "")).strip()
    long_name = str(row.get("route_long_name", "")).strip()

    if short_name and short_name.lower() != "nan" and long_name and long_name.lower() != "nan":
        return f"{short_name} — {long_name} [{route_id}]"
    if short_name and short_name.lower() != "nan":
        return f"{short_name} [{route_id}]"
    if long_name and long_name.lower() != "nan":
        return f"{long_name} [{route_id}]"
    return route_id


def _aggregate_all_directions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    work = df.copy()

    agg = work.groupby("snapshot_time", as_index=False).agg(
        n_buses=("n_buses", "sum"),
        delayed=("delayed", "sum"),
        avg_delay_min=("avg_delay_min", "mean"),
        p95_delay_min=("p95_delay_min", "mean"),
        min_delay_min=("min_delay_min", "min"),
        max_delay_min=("max_delay_min", "max"),
    )

    agg["pct_delayed"] = 0.0
    mask = agg["n_buses"].fillna(0) > 0
    agg.loc[mask, "pct_delayed"] = 100.0 * agg.loc[mask, "delayed"] / agg.loc[mask, "n_buses"]

    return agg.sort_values("snapshot_time").reset_index(drop=True)


def _make_metric(value, digits: int = 1, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def _bus_color(delay_sec) -> str:
    if delay_sec is None or pd.isna(delay_sec):
        return "gray"
    if delay_sec > 60:
        return "red"
    if delay_sec < -60:
        return "blue"
    return "green"


def _bus_status_label(delay_sec) -> str:
    if delay_sec is None or pd.isna(delay_sec):
        return "unknown"
    if delay_sec > 60:
        return "delayed"
    if delay_sec < -60:
        return "early"
    return "on time"


def _prepare_live_route_vehicles(vehicle_df: pd.DataFrame, selected_route: str, selected_direction: str | None):
    if vehicle_df.empty:
        return vehicle_df

    work = vehicle_df.copy()

    if "route_id" in work.columns:
        work["route_id"] = work["route_id"].astype(str).str.strip()

    if "direction_id" in work.columns:
        work["direction_id"] = work["direction_id"].apply(_normalize_direction_id)

    work = work[work["route_id"] == str(selected_route).strip()].copy()

    if selected_direction is not None:
        work = work[work["direction_id"] == _normalize_direction_id(selected_direction)].copy()

    if work.empty:
        return work

    if "current_status" in work.columns:
        work["status_text"] = work["current_status"].apply(decode_vehicle_status)
    else:
        work["status_text"] = None

    if "delay_sec" in work.columns:
        work["delay_min"] = pd.to_numeric(work["delay_sec"], errors="coerce") / 60.0
    else:
        work["delay_min"] = pd.NA

    work["delay_color"] = work["delay_sec"].apply(_bus_color) if "delay_sec" in work.columns else "gray"
    work["delay_state"] = work["delay_sec"].apply(_bus_status_label) if "delay_sec" in work.columns else "unknown"

    return work


def _render_route_map(route_lines_gdf, route_stops_gdf, live_route_vehicles: pd.DataFrame, map_key: str):
    if route_lines_gdf.empty and route_stops_gdf.empty and live_route_vehicles.empty:
        st.info("No route geometry or live vehicles available.")
        return

    if not route_lines_gdf.empty:
        center_geom = route_lines_gdf.geometry.iloc[0].centroid
        center = [center_geom.y, center_geom.x]
    elif not route_stops_gdf.empty:
        center = [route_stops_gdf.geometry.iloc[0].y, route_stops_gdf.geometry.iloc[0].x]
    else:
        center = [live_route_vehicles["latitude"].iloc[0], live_route_vehicles["longitude"].iloc[0]]

    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    if not route_lines_gdf.empty:
        for _, row in route_lines_gdf.iterrows():
            direction = row.get("direction_id")
            trips = row.get("n_trips")
            tooltip = f"Direction: {direction if direction is not None else 'n/a'} | representative shape | trips: {trips}"
            folium.GeoJson(
                row.geometry.__geo_interface__,
                tooltip=tooltip,
                style_function=lambda _x: {
                    "color": "#d62728",
                    "weight": 5,
                    "opacity": 0.85,
                },
            ).add_to(m)

        bounds = route_lines_gdf.total_bounds
        if bounds is not None and len(bounds) == 4:
            minx, miny, maxx, maxy = bounds
            m.fit_bounds([[miny, minx], [maxy, maxx]])

    if not route_stops_gdf.empty:
        for _, row in route_stops_gdf.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=3,
                weight=1,
                color="black",
                fill=True,
                fill_color="white",
                fill_opacity=0.8,
                popup=(
                    f"{row.get('stop_name', '')}<br>"
                    f"stop_id: {row.get('stop_id', '')}<br>"
                    f"seq: {row.get('stop_sequence', '')}<br>"
                    f"dir: {row.get('direction_id', '')}"
                ),
            ).add_to(m)

    if not live_route_vehicles.empty:
        for _, row in live_route_vehicles.iterrows():
            vehicle_label = row.get("vehicle_label") if pd.notna(row.get("vehicle_label")) else row.get("vehicle_id")
            delay_min = row.get("delay_min")
            delay_txt = "n/a" if pd.isna(delay_min) else f"{delay_min:.1f} min"
            popup = (
                f"Vehicle: {vehicle_label}<br>"
                f"Trip: {row.get('trip_id', '')}<br>"
                f"Direction: {row.get('direction_id', '')}<br>"
                f"Status: {row.get('delay_state', 'unknown')}<br>"
                f"Delay: {delay_txt}<br>"
                f"Stop: {row.get('stop_id', '')}<br>"
                f"Current status: {row.get('status_text', '')}"
            )

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=7,
                weight=2,
                color=row.get("delay_color", "gray"),
                fill=True,
                fill_color=row.get("delay_color", "gray"),
                fill_opacity=0.95,
                popup=popup,
                tooltip=f"{vehicle_label} | {row.get('delay_state', 'unknown')}",
            ).add_to(m)

    st_folium(m, use_container_width=True, height=500, key=map_key)


def render_route_performance_view(gtfs_static_path: str):
    st.subheader("Route performance")

    routes_df, _ = load_gtfs_static(gtfs_static_path)
    if routes_df.empty:
        st.warning("Static GTFS routes could not be loaded. Check GTFS_STATIC_PATH.")
        return

    routes_df = routes_df.copy()
    routes_df["route_id"] = routes_df["route_id"].astype(str).str.strip()
    routes_df["route_display"] = routes_df.apply(_build_route_label, axis=1)
    routes_df = routes_df.sort_values("route_display").reset_index(drop=True)

    route_options = routes_df["route_display"].tolist()
    selected_route_display = st.selectbox("Route", route_options, key="route_perf_route")

    selected_route_row = routes_df.loc[routes_df["route_display"] == selected_route_display].iloc[0]
    selected_route = str(selected_route_row["route_id"]).strip()

    hist_all = load_gtfs_route_last_24h(selected_route, None)
    if hist_all.empty:
        st.info("No historical data available for the selected route in the last 24 hours.")
        return

    if "direction_id" in hist_all.columns:
        hist_all["direction_id"] = hist_all["direction_id"].apply(_normalize_direction_id)

    direction_values = []
    if "direction_id" in hist_all.columns:
        direction_values = sorted(
            [d for d in hist_all["direction_id"].dropna().unique().tolist() if str(d).strip() != ""]
        )

    direction_options = ["All"] + direction_values
    selected_direction = st.selectbox("Direction", direction_options, index=0, key="route_perf_direction")

    if selected_direction == "All":
        hist_df = _aggregate_all_directions(hist_all)
        
        direction_for_map = None
    else:
        hist_df = hist_all[hist_all["direction_id"] == selected_direction].copy()
        hist_df = hist_df.sort_values("snapshot_time").reset_index(drop=True)
        direction_for_map = selected_direction

    hist_df = _clean_route_delay_metrics(hist_df)
    if hist_df.empty:
        st.info("No historical data available for the selected route/direction in the last 24 hours.")
        return

    # live vehicles for this route
    live_vehicle_df = fetch_gtfs_rt_vehicle_positions(VEHICLE_URL)
    if not live_vehicle_df.empty:
        live_vehicle_df["route_id"] = live_vehicle_df["route_id"].astype(str).str.strip()
    live_vehicle_df = join_vehicle_positions_with_routes(live_vehicle_df, routes_df)

    trip_updates_df = fetch_gtfs_rt_trip_updates(TRIP_UPDATES_URL)
    if not live_vehicle_df.empty and not trip_updates_df.empty:
        live_vehicle_df = enrich_vehicle_positions_with_delay(live_vehicle_df, trip_updates_df)
    else:
        if "delay_sec" not in live_vehicle_df.columns:
            live_vehicle_df["delay_sec"] = pd.NA

    live_route_vehicles = _prepare_live_route_vehicles(
        live_vehicle_df,
        selected_route,
        None if selected_direction == "All" else selected_direction,
    )

    latest = hist_df.sort_values("snapshot_time").iloc[-1]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Buses", f"{int(latest['n_buses'])}" if pd.notna(latest.get("n_buses")) else "n/a")
    c2.metric("Delayed", f"{int(latest['delayed'])}" if pd.notna(latest.get("delayed")) else "n/a")
    c3.metric("Delayed %", _make_metric(latest.get("pct_delayed"), 1, "%"))
    c4.metric("Avg delay (min)", _make_metric(latest.get("avg_delay_min"), 2))
    c5.metric("Live buses now", str(len(live_route_vehicles)))

    left, right = st.columns([1.2, 1.1])

    with left:
        st.markdown("#### Last 24h trend")

        plot_df = hist_df.copy()
        plot_df = plot_df.sort_values("snapshot_time").set_index("snapshot_time")

        if {"n_buses", "delayed"}.issubset(plot_df.columns):
            st.markdown("**Fleet and delayed vehicles**")
            st.line_chart(plot_df[["n_buses", "delayed"]], height=220)

        if "pct_delayed" in plot_df.columns:
            st.markdown("**Percentage of delayed vehicles**")
            st.line_chart(plot_df[["pct_delayed"]], height=180)

        if "avg_delay_min" in plot_df.columns:
            st.markdown("**Average delay (minutes)**")
            st.line_chart(plot_df[["avg_delay_min"]], height=180)

    with right:
        st.markdown("#### Route map + live vehicles")
        route_lines_gdf, route_stops_gdf = load_gtfs_route_map_data(
            gtfs_static_path,
            selected_route,
            direction_for_map,
        )
        _render_route_map(
            route_lines_gdf,
            route_stops_gdf,
            live_route_vehicles,
            map_key=f"route_map_{selected_route}_{selected_direction}",
        )

        st.caption("Bus colors: red = delayed, green = on time, blue = early, gray = unknown.")

    with st.expander("Live vehicles on this route", expanded=False):
        if live_route_vehicles.empty:
            st.write("No live vehicles available for this route right now.")
        else:
            cols = [
                c for c in [
                    "route_label",
                    "route_id",
                    "direction_id",
                    "trip_id",
                    "vehicle_id",
                    "vehicle_label",
                    "status_text",
                    "delay_state",
                    "delay_sec",
                    "delay_min",
                    "stop_id",
                    "timestamp_dt",
                ] if c in live_route_vehicles.columns
            ]
            st.dataframe(
                live_route_vehicles[cols].sort_values(
                    ["direction_id", "delay_sec"],
                    ascending=[True, False],
                    na_position="last",
                ),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("Show latest historical records", expanded=False):
        show_cols = [
            c for c in [
                "snapshot_time",
                "n_buses",
                "delayed",
                "pct_delayed",
                "avg_delay_min",
                "p95_delay_min",
                "min_delay_min",
                "max_delay_min",
            ]
            if c in hist_df.columns
        ]
        st.dataframe(
            hist_df[show_cols].sort_values("snapshot_time", ascending=False).head(180),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("System-wide GTFS summary", expanded=False):
        if live_vehicle_df.empty:
            st.write("No system-wide live GTFS data available.")
        else:
            overall_df = build_overall_gtfs_stats(live_vehicle_df)
            st.dataframe(overall_df, use_container_width=True, hide_index=True)
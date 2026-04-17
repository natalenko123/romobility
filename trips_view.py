import json

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from branca.colormap import linear
from shapely.geometry import LineString
from streamlit_folium import st_folium

from config import DEFAULT_H3_RESOLUTION, DEFAULT_MAX_TRIPS_TO_DRAW, DEFAULT_TRIPS_SELECTED_ON_MAP, MAP_CENTER, MAP_ZOOM
from loaders import load_trip_points_for_selected_ids, load_trips_with_quartieri
from map_utils import add_focus_css, add_zone_outline, get_zone_from_click
from ui_helpers import update_zone_from_click


def filter_trip_ids_between_quartieri(
    trips_q: pd.DataFrame,
    origin_zone: str,
    destination_zone: str,
) -> pd.DataFrame:
    return trips_q.loc[
        (trips_q["origin_quartiere"] == origin_zone)
        & (trips_q["destination_quartiere"] == destination_zone),
        ["trip_id", "origin_quartiere", "destination_quartiere"],
    ].drop_duplicates().copy()


def select_trip_ids_for_display(
    trip_ids_df: pd.DataFrame,
    max_trips: int,
    sample_mode: str = "first",
) -> pd.DataFrame:
    trip_ids_df = trip_ids_df.drop_duplicates("trip_id").copy()
    if len(trip_ids_df) <= max_trips:
        return trip_ids_df
    if sample_mode == "random":
        return trip_ids_df.sample(n=max_trips, random_state=42)
    return trip_ids_df.head(max_trips)


def build_trip_lines(trip_points: pd.DataFrame, trip_ids_df: pd.DataFrame) -> gpd.GeoDataFrame:
    if trip_points.empty:
        return gpd.GeoDataFrame(
            columns=["trip_id", "n_points", "travel_time_min", "length_km", "mean_speed_kmh", "geometry", "origin_quartiere", "destination_quartiere"],
            geometry="geometry",
            crs=4326,
        )

    pts = trip_points.sort_values(["trip_id", "ts"]).copy()
    rows = []

    for trip_id, grp in pts.groupby("trip_id", sort=False):
        coords = list(zip(grp["longitude"], grp["latitude"]))
        if len(coords) < 2:
            continue

        t0 = grp["ts"].min()
        t1 = grp["ts"].max()
        travel_time_min = (t1 - t0).total_seconds() / 60.0
        line = LineString(coords)
        line_len = gpd.GeoSeries([line], crs=4326).to_crs(3857).length.iloc[0] / 1000.0
        mean_speed_kmh = np.nan if travel_time_min <= 0 else line_len / (travel_time_min / 60.0)

        rows.append(
            {
                "trip_id": trip_id,
                "n_points": len(coords),
                "travel_time_min": round(travel_time_min, 2),
                "length_km": round(float(line_len), 3),
                "mean_speed_kmh": round(float(mean_speed_kmh), 2) if pd.notna(mean_speed_kmh) else np.nan,
                "geometry": line,
            }
        )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
    if not trip_ids_df.empty and not gdf.empty:
        gdf = gdf.merge(trip_ids_df.drop_duplicates("trip_id"), on="trip_id", how="left")
    return gdf


def clip_trip_points_to_od_segment(
    trip_points: pd.DataFrame,
    zones: gpd.GeoDataFrame,
    zone_field: str,
    origin_zone: str,
    destination_zone: str,
) -> pd.DataFrame:
    if trip_points.empty:
        return pd.DataFrame(columns=list(trip_points.columns) + ["od_segment"])

    pts = trip_points.sort_values(["trip_id", "ts"]).copy().reset_index(drop=True)
    gpts = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts["longitude"], pts["latitude"]), crs=4326)

    origin_geom = zones.loc[zones[zone_field].astype(str).str.strip() == str(origin_zone).strip(), "geometry"]
    destination_geom = zones.loc[zones[zone_field].astype(str).str.strip() == str(destination_zone).strip(), "geometry"]

    if origin_geom.empty or destination_geom.empty:
        return pd.DataFrame(columns=list(trip_points.columns) + ["od_segment"])

    origin_union = origin_geom.iloc[0]
    dest_union = destination_geom.iloc[0]

    gpts["in_origin"] = gpts.geometry.within(origin_union) | gpts.geometry.intersects(origin_union)
    gpts["in_destination"] = gpts.geometry.within(dest_union) | gpts.geometry.intersects(dest_union)

    kept_parts = []
    for _, grp in gpts.groupby("trip_id", sort=False):
        grp = grp.sort_values("ts").copy()
        origin_positions = np.flatnonzero(grp["in_origin"].to_numpy())
        dest_positions = np.flatnonzero(grp["in_destination"].to_numpy())

        if len(origin_positions) == 0 or len(dest_positions) == 0:
            continue

        last_origin_pos = int(origin_positions[-1])
        first_dest_pos = int(dest_positions[0])

        if last_origin_pos >= first_dest_pos:
            continue

        seg = grp.iloc[last_origin_pos:first_dest_pos + 1].copy()
        if len(seg) < 2:
            continue

        seg["od_segment"] = True
        kept_parts.append(seg)

    if not kept_parts:
        return pd.DataFrame(columns=list(trip_points.columns) + ["od_segment"])

    out = pd.concat(kept_parts, ignore_index=True)
    return pd.DataFrame(out.drop(columns="geometry"))


def build_clipped_trip_lines(clipped_trip_points: pd.DataFrame, trip_ids_df: pd.DataFrame) -> gpd.GeoDataFrame:
    return build_trip_lines(clipped_trip_points, trip_ids_df)


def make_od_summary(
    trip_ids_df: pd.DataFrame,
    full_trip_lines_gdf: gpd.GeoDataFrame,
    clipped_trip_lines_gdf: gpd.GeoDataFrame,
    hour_min: int,
    hour_max: int,
):
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("OD trips", len(trip_ids_df))
    c2.metric("Hour window", f"{hour_min:02d}-{hour_max:02d}")

    if full_trip_lines_gdf.empty:
        c3.metric("Full median TT", "—")
        c4.metric("Full mean len", "—")
    else:
        c3.metric("Full median TT (min)", round(full_trip_lines_gdf["travel_time_min"].median(), 2))
        c4.metric("Full mean len (km)", round(full_trip_lines_gdf["length_km"].mean(), 2))

    if clipped_trip_lines_gdf.empty:
        c5.metric("Clipped kept", 0)
        c6.metric("Clipped median TT", "—")
    else:
        c5.metric("Clipped kept", len(clipped_trip_lines_gdf))
        c6.metric("Clipped median TT (min)", round(clipped_trip_lines_gdf["travel_time_min"].median(), 2))


def make_trip_lines_map(
    zones: gpd.GeoDataFrame,
    zone_field: str,
    origin_zone: str,
    destination_zone: str,
    trip_lines_gdf: gpd.GeoDataFrame,
    line_mode_label: str,
) -> folium.Map:
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles="cartodbpositron")
    add_focus_css(m)

    base_gdf = zones[[zone_field, "geometry"]].copy()
    folium.GeoJson(
        json.loads(base_gdf.to_json()),
        name="quartieri",
        style_function=lambda _: {"fillColor": "#f2f2f2", "color": "#999999", "weight": 0.8, "fillOpacity": 0.18},
        tooltip=folium.GeoJsonTooltip(fields=[zone_field], aliases=["Quartiere"], sticky=False),
    ).add_to(m)

    add_zone_outline(m, zones, zone_field, origin_zone, color="#000000", weight=3, dash_array="8, 8", fill_color="#000000", fill_opacity=0.06)
    add_zone_outline(m, zones, zone_field, destination_zone, color="#cc0000", weight=3, fill_color="#cc0000", fill_opacity=0.06)

    if not trip_lines_gdf.empty:
        vals = trip_lines_gdf["travel_time_min"].dropna()
        if len(vals) > 0:
            vmin = float(vals.min())
            vmax = float(vals.max())
            if vmin == vmax:
                vmax = vmin + 1e-6
            colormap = linear.YlOrRd_09.scale(vmin, vmax)
        else:
            colormap = linear.YlOrRd_09.scale(0, 1)
        colormap.caption = f"{line_mode_label} travel time (min)"

        draw_gdf = trip_lines_gdf[["trip_id", "origin_quartiere", "destination_quartiere", "n_points", "travel_time_min", "length_km", "mean_speed_kmh", "geometry"]].copy()

        def trip_style_function(feature):
            val = feature["properties"].get("travel_time_min")
            color = "#666666" if val is None else colormap(val)
            return {"color": color, "weight": 3, "opacity": 0.65}

        folium.GeoJson(
            json.loads(draw_gdf.to_json()),
            name="trips",
            style_function=trip_style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=["trip_id", "origin_quartiere", "destination_quartiere", "travel_time_min", "length_km", "mean_speed_kmh", "n_points"],
                aliases=["Trip ID", "Origin", "Destination", f"{line_mode_label} TT (min)", "Length (km)", "Mean speed (km/h)", "Points"],
                sticky=False,
                labels=True,
            ),
        ).add_to(m)

        colormap.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def render_trips_view(zones, zone_field, choices, trips_with_quartieri_path: str, trip_points_path: str):
    with st.sidebar:
        st.selectbox("Origin quartiere", choices, key="shared_origin")
        st.selectbox("Destination quartiere", choices, key="shared_destination")
        trip_line_mode = st.radio("Trip geometry", ["Full trips", "OD-clipped trips"], index=0, key="trip_line_mode")
        max_trips_to_draw = st.slider("Maximum trips to draw", 10, 1000, DEFAULT_MAX_TRIPS_TO_DRAW, 10)
        sample_mode = st.radio("Trip selection mode", ["first", "random"], index=0, key="trip_sample_mode")

    hour_min, hour_max = st.session_state["shared_hour_range"]
    origin_zone = st.session_state["shared_origin"]
    destination_zone = st.session_state["shared_destination"]

    trips_q = load_trips_with_quartieri(trips_with_quartieri_path)

    trip_ids_df = filter_trip_ids_between_quartieri(trips_q, origin_zone, destination_zone)
    selected_trip_ids_df = select_trip_ids_for_display(trip_ids_df, max_trips_to_draw, sample_mode)
    selected_trip_ids = tuple(selected_trip_ids_df["trip_id"].astype(str).tolist())

    trip_points = load_trip_points_for_selected_ids(
        parquet_path=str(trip_points_path),
        selected_trip_ids=selected_trip_ids,
        h3_resolution=DEFAULT_H3_RESOLUTION,
        hour_min=hour_min,
        hour_max=hour_max,
    )

    full_trip_lines_gdf = build_trip_lines(trip_points, selected_trip_ids_df)
    clipped_points = clip_trip_points_to_od_segment(trip_points, zones, zone_field, origin_zone, destination_zone)
    clipped_trip_lines_gdf = build_clipped_trip_lines(clipped_points, selected_trip_ids_df)

    make_od_summary(selected_trip_ids_df, full_trip_lines_gdf, clipped_trip_lines_gdf, hour_min, hour_max)

    trip_lines_source = full_trip_lines_gdf if trip_line_mode == "Full trips" else clipped_trip_lines_gdf

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Origin", origin_zone)
    c2.metric("Destination", destination_zone)
    c3.metric("Matching trips", len(trip_ids_df))
    c4.metric("Trips available to display", len(trip_lines_source))

    available_trip_ids = trip_lines_source["trip_id"].tolist() if not trip_lines_source.empty else []
    selected_trip_ids_on_map = st.multiselect(
        "Select trips to display",
        options=available_trip_ids,
        default=available_trip_ids[: min(DEFAULT_TRIPS_SELECTED_ON_MAP, len(available_trip_ids))],
    )

    if selected_trip_ids_on_map:
        trip_lines_display = trip_lines_source.loc[trip_lines_source["trip_id"].isin(selected_trip_ids_on_map)].copy()
    else:
        trip_lines_display = trip_lines_source.iloc[0:0].copy()

    if not trip_lines_display.empty:
        s1, s2, s3 = st.columns(3)
        s1.metric("Mean travel time (min)", round(trip_lines_display["travel_time_min"].mean(), 2))
        s2.metric("Mean length (km)", round(trip_lines_display["length_km"].mean(), 2))
        s3.metric("Mean speed (km/h)", round(trip_lines_display["mean_speed_kmh"].mean(), 2))

    m = make_trip_lines_map(
        zones,
        zone_field,
        origin_zone,
        destination_zone,
        trip_lines_display,
        "Full" if trip_line_mode == "Full trips" else "Clipped",
    )
    trip_map_data = st_folium(m, width=None, height=760)

    clicked_trip_zone = None
    if trip_map_data and trip_map_data.get("last_clicked"):
        clicked_trip_zone = get_zone_from_click(zones, zone_field, trip_map_data["last_clicked"]["lat"], trip_map_data["last_clicked"]["lng"])

    update_zone_from_click(
        clicked_trip_zone,
        "_pending_shared_origin",
        "_pending_shared_destination",
        "trip_use_origin",
        "trip_use_destination",
    )

    with st.expander("Matching trip IDs"):
        st.dataframe(trip_ids_df[["trip_id", "origin_quartiere", "destination_quartiere"]], use_container_width=True)

    with st.expander("Trip statistics table"):
        if trip_lines_display.empty:
            st.write("No trips selected for display.")
        else:
            st.dataframe(
                trip_lines_display[
                    ["trip_id", "origin_quartiere", "destination_quartiere", "travel_time_min", "length_km", "mean_speed_kmh", "n_points"]
                ].sort_values("travel_time_min", ascending=False),
                use_container_width=True,
            )

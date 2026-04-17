import json

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from branca.colormap import linear
from h3 import cell_to_boundary
from shapely.geometry import LineString, Polygon
from streamlit_folium import st_folium

from config import (
    DEFAULT_H3_RESOLUTION,
    DEFAULT_MAX_TRIPS_FOR_CORRIDOR,
    MAP_CENTER,
    MAP_ZOOM,
    OSM_ROADS_PATH,
)
from loaders import (
    load_osm_roads_for_rome,
    load_trip_points_for_selected_ids,
    load_trips_with_quartieri,
)
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
            columns=[
                "trip_id",
                "n_points",
                "travel_time_min",
                "length_km",
                "mean_speed_kmh",
                "geometry",
                "origin_quartiere",
                "destination_quartiere",
            ],
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

        geom = LineString(coords)
        line_len = gpd.GeoSeries([geom], crs=4326).to_crs(3857).length.iloc[0] / 1000.0
        mean_speed_kmh = np.nan if travel_time_min <= 0 else line_len / (travel_time_min / 60.0)

        rows.append(
            {
                "trip_id": trip_id,
                "n_points": len(coords),
                "travel_time_min": round(travel_time_min, 2),
                "length_km": round(float(line_len), 3),
                "mean_speed_kmh": round(float(mean_speed_kmh), 2) if pd.notna(mean_speed_kmh) else np.nan,
                "geometry": geom,
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
    gpts = gpd.GeoDataFrame(
        pts,
        geometry=gpd.points_from_xy(pts["longitude"], pts["latitude"]),
        crs=4326,
    )

    origin_geom = zones.loc[
        zones[zone_field].astype(str).str.strip() == str(origin_zone).strip(),
        "geometry",
    ]
    destination_geom = zones.loc[
        zones[zone_field].astype(str).str.strip() == str(destination_zone).strip(),
        "geometry",
    ]

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


def build_clipped_trip_lines(
    clipped_trip_points: pd.DataFrame,
    trip_ids_df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    return build_trip_lines(clipped_trip_points, trip_ids_df)


def summarize_h3_usage(clipped_trip_points: pd.DataFrame) -> pd.DataFrame:
    if clipped_trip_points.empty:
        return pd.DataFrame(columns=["h3_cell", "n_trips"])

    return (
        clipped_trip_points[["trip_id", "h3_cell"]]
        .drop_duplicates()
        .groupby("h3_cell", as_index=False)
        .agg(n_trips=("trip_id", "nunique"))
        .sort_values("n_trips", ascending=False)
        .reset_index(drop=True)
    )


def h3_usage_to_gdf(h3_usage_df: pd.DataFrame) -> gpd.GeoDataFrame:
    if h3_usage_df.empty:
        return gpd.GeoDataFrame(columns=["h3_cell", "n_trips", "geometry"], geometry="geometry", crs=4326)

    rows = []
    for cell, n_trips in h3_usage_df[["h3_cell", "n_trips"]].itertuples(index=False):
        boundary = cell_to_boundary(cell)
        poly = Polygon([(lng, lat) for lat, lng in boundary])
        rows.append({"h3_cell": cell, "n_trips": n_trips, "geometry": poly})

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)


def assign_clipped_points_to_roads(
    clipped_trip_points: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    if clipped_trip_points.empty:
        return pd.DataFrame(columns=list(clipped_trip_points.columns) + ["u", "v", "key", "street_name", "dist_to_road_m"])

    pts = clipped_trip_points.copy()
    gpts = gpd.GeoDataFrame(
        pts,
        geometry=gpd.points_from_xy(pts["longitude"], pts["latitude"]),
        crs=4326,
    )

    gpts_m = gpts.to_crs(3857)
    roads_m = roads_gdf.to_crs(3857)

    joined = gpd.sjoin_nearest(
        gpts_m,
        roads_m[["u", "v", "key", "street_name", "geometry"]],
        how="left",
        distance_col="dist_to_road_m",
    )

    return pd.DataFrame(joined.drop(columns="geometry"))


def summarize_used_road_pieces(
    clipped_points_with_roads: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if clipped_points_with_roads.empty:
        return gpd.GeoDataFrame(
            columns=["u", "v", "key", "street_name", "n_trips", "n_points", "geometry"],
            geometry="geometry",
            crs=4326,
        )

    df = clipped_points_with_roads.copy()
    df = df[df["street_name"].notna()]
    df = df[df["street_name"].astype(str).str.strip() != ""]
    df = df[df["street_name"] != "Unnamed road"]

    if df.empty:
        return gpd.GeoDataFrame(
            columns=["u", "v", "key", "street_name", "n_trips", "n_points", "geometry"],
            geometry="geometry",
            crs=4326,
        )

    trip_counts = (
        df[["trip_id", "u", "v", "key", "street_name"]]
        .drop_duplicates()
        .groupby(["u", "v", "key", "street_name"], as_index=False)
        .agg(n_trips=("trip_id", "nunique"))
    )

    point_counts = (
        df[["u", "v", "key"]]
        .groupby(["u", "v", "key"], as_index=False)
        .size()
        .rename(columns={"size": "n_points"})
    )

    usage = trip_counts.merge(point_counts, on=["u", "v", "key"], how="left")

    used_edges = roads_gdf.merge(
        usage,
        on=["u", "v", "key", "street_name"],
        how="inner",
    )

    return used_edges.sort_values(["n_trips", "n_points"], ascending=False).reset_index(drop=True)


def summarize_street_usage_from_used_edges(used_edges_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    if used_edges_gdf.empty:
        return pd.DataFrame(columns=["street_name", "n_trips", "n_segments"])

    return (
        used_edges_gdf.groupby("street_name", as_index=False)
        .agg(n_trips=("n_trips", "sum"), n_segments=("geometry", "size"))
        .sort_values(["n_trips", "n_segments"], ascending=False)
        .reset_index(drop=True)
    )


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


def make_corridor_usage_map(
    zones: gpd.GeoDataFrame,
    zone_field: str,
    origin_zone: str,
    destination_zone: str,
    h3_usage_gdf: gpd.GeoDataFrame,
) -> folium.Map:
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles="cartodbpositron")
    add_focus_css(m)

    base_gdf = zones[[zone_field, "geometry"]].copy()
    folium.GeoJson(
        json.loads(base_gdf.to_json()),
        name="quartieri",
        style_function=lambda _: {
            "fillColor": "#f2f2f2",
            "color": "#999999",
            "weight": 0.8,
            "fillOpacity": 0.15,
        },
        tooltip=folium.GeoJsonTooltip(fields=[zone_field], aliases=["Quartiere"], sticky=False),
    ).add_to(m)

    add_zone_outline(
        m,
        zones,
        zone_field,
        origin_zone,
        color="#000000",
        weight=3,
        dash_array="8, 8",
        fill_color="#000000",
        fill_opacity=0.06,
    )
    add_zone_outline(
        m,
        zones,
        zone_field,
        destination_zone,
        color="#cc0000",
        weight=3,
        fill_color="#cc0000",
        fill_opacity=0.06,
    )

    if not h3_usage_gdf.empty:
        vals = h3_usage_gdf["n_trips"].dropna()
        if len(vals) > 0:
            vmin = float(vals.min())
            vmax = float(vals.max())
            if vmin == vmax:
                vmax = vmin + 1e-6
            colormap = linear.YlOrRd_09.scale(vmin, vmax)
        else:
            colormap = linear.YlOrRd_09.scale(0, 1)
        colormap.caption = "Trips passing through cell"

        def h3_style(feature):
            val = feature["properties"].get("n_trips")
            if val is None:
                return {"fillColor": "#d9d9d9", "color": "#666666", "weight": 0.6, "fillOpacity": 0.15}
            return {"fillColor": colormap(val), "color": "#666666", "weight": 0.6, "fillOpacity": 0.60}

        folium.GeoJson(
            json.loads(h3_usage_gdf[["h3_cell", "n_trips", "geometry"]].to_json()),
            name="corridor_usage",
            style_function=h3_style,
            tooltip=folium.GeoJsonTooltip(
                fields=["h3_cell", "n_trips"],
                aliases=["H3 cell", "Trips"],
                sticky=False,
                labels=True,
            ),
        ).add_to(m)

        colormap.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def render_corridor_view(
    zones,
    zone_field,
    choices,
    trips_with_quartieri_path: str,
    trip_points_path: str,
):
    with st.sidebar:
        st.selectbox("Origin quartiere", choices, key="shared_origin")
        st.selectbox("Destination quartiere", choices, key="shared_destination")
        max_trips_for_corridor = st.slider(
            "Maximum trips to analyze",
            20,
            2000,
            DEFAULT_MAX_TRIPS_FOR_CORRIDOR,
            20,
        )
        sample_mode_corridor = st.radio(
            "Trip selection mode",
            ["first", "random"],
            index=1,
            key="corr_sample_mode",
        )
        h3_resolution = st.slider("H3 cell resolution", 8, 10, DEFAULT_H3_RESOLUTION, 1)
        top_n_streets = st.slider(
            "Top streets in results table",
            5,
            50,
            15,
            1,
            key="corr_top_streets",
        )

    hour_min, hour_max = st.session_state["shared_hour_range"]
    origin_zone = st.session_state["shared_origin"]
    destination_zone = st.session_state["shared_destination"]

    trips_q = load_trips_with_quartieri(trips_with_quartieri_path)

    trip_ids_df = filter_trip_ids_between_quartieri(trips_q, origin_zone, destination_zone)
    selected_trip_ids_df = select_trip_ids_for_display(
        trip_ids_df,
        max_trips_for_corridor,
        sample_mode_corridor,
    )
    selected_trip_ids = tuple(selected_trip_ids_df["trip_id"].astype(str).tolist())

    trip_points = load_trip_points_for_selected_ids(
        parquet_path=str(trip_points_path),
        selected_trip_ids=selected_trip_ids,
        h3_resolution=h3_resolution,
        hour_min=hour_min,
        hour_max=hour_max,
    )

    full_trip_lines_gdf = build_trip_lines(trip_points, selected_trip_ids_df)
    clipped_points = clip_trip_points_to_od_segment(
        trip_points,
        zones,
        zone_field,
        origin_zone,
        destination_zone,
    )
    clipped_trip_lines_gdf = build_clipped_trip_lines(clipped_points, selected_trip_ids_df)

    make_od_summary(selected_trip_ids_df, full_trip_lines_gdf, clipped_trip_lines_gdf, hour_min, hour_max)

    h3_usage_df = summarize_h3_usage(clipped_points)
    h3_usage_gdf = h3_usage_to_gdf(h3_usage_df)

    roads_gdf = load_osm_roads_for_rome(str(OSM_ROADS_PATH))
    clipped_points_with_roads = assign_clipped_points_to_roads(clipped_points, roads_gdf)
    used_edges_gdf = summarize_used_road_pieces(clipped_points_with_roads, roads_gdf)
    street_usage_df = summarize_street_usage_from_used_edges(used_edges_gdf)

    m = make_corridor_usage_map(
        zones,
        zone_field,
        origin_zone,
        destination_zone,
        h3_usage_gdf,
    )
    corr_map_data = st_folium(m, width=None, height=760)

    clicked_corr_zone = None
    if corr_map_data and corr_map_data.get("last_clicked"):
        clicked_corr_zone = get_zone_from_click(
            zones,
            zone_field,
            corr_map_data["last_clicked"]["lat"],
            corr_map_data["last_clicked"]["lng"],
        )

    update_zone_from_click(
        clicked_corr_zone,
        "_pending_shared_origin",
        "_pending_shared_destination",
        "corr_use_origin",
        "corr_use_destination",
    )

    with st.expander("Most used H3 cells"):
        if h3_usage_df.empty:
            st.write("No H3 usage found.")
        else:
            st.dataframe(h3_usage_df.head(200), use_container_width=True)

    with st.expander("Most used streets in this corridor"):
        if street_usage_df.empty:
            st.write("No street usage found.")
        else:
            st.dataframe(street_usage_df.head(top_n_streets), use_container_width=True)

    with st.expander("Clipped trip statistics"):
        if clipped_trip_lines_gdf.empty:
            st.write("No clipped trips found.")
        else:
            st.dataframe(
                clipped_trip_lines_gdf[
                    [
                        "trip_id",
                        "origin_quartiere",
                        "destination_quartiere",
                        "travel_time_min",
                        "length_km",
                        "mean_speed_kmh",
                        "n_points",
                    ]
                ].sort_values("travel_time_min", ascending=False),
                use_container_width=True,
            )

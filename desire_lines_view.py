import json

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import LineString
from streamlit_folium import st_folium

from config import MAP_CENTER, MAP_ZOOM
from loaders import load_od, load_trip_hours_for_pairs, load_trips_with_quartieri
from map_utils import (
    add_focus_css,
    add_zone_outline,
    get_zone_from_click,
    make_high_contrast_trip_colormap,
)
from ui_helpers import update_zone_from_click


def build_demo_od(zones: gpd.GeoDataFrame, zone_field: str) -> pd.DataFrame:
    z = zones[[zone_field, "geometry"]].copy()
    z[zone_field] = z[zone_field].astype(str).str.strip()

    z_proj = z.to_crs(3857)
    cent = gpd.GeoSeries(z_proj.geometry.centroid, crs=3857).to_crs(4326)

    z["cx"] = cent.x
    z["cy"] = cent.y

    a = z[[zone_field, "cx", "cy"]].rename(columns={zone_field: "origin_quartiere", "cx": "ox", "cy": "oy"})
    b = z[[zone_field, "cx", "cy"]].rename(columns={zone_field: "destination_quartiere", "cx": "dx", "cy": "dy"})
    od = a.merge(b, how="cross")

    dist = np.sqrt((od["ox"] - od["dx"]) ** 2 + (od["oy"] - od["dy"]) ** 2)
    rng = np.random.default_rng(42)

    base = 6 + dist * 180
    noise = rng.normal(0, 2.0, size=len(base))
    median_tt = np.clip(base + noise, 3, None)
    mean_tt = median_tt * rng.uniform(1.0, 1.08, size=len(base))
    p85_tt = median_tt * rng.uniform(1.15, 1.35, size=len(base))
    min_tt = np.clip(median_tt * rng.uniform(0.65, 0.9, size=len(base)), 1, None)
    max_tt = p85_tt * rng.uniform(1.1, 1.4, size=len(base))
    n_trips = rng.integers(3, 80, size=len(base))

    return pd.DataFrame(
        {
            "origin_quartiere": od["origin_quartiere"],
            "destination_quartiere": od["destination_quartiere"],
            "n_trips": n_trips,
            "median_tt_min": median_tt.round(1),
            "mean_tt_min": mean_tt.round(1),
            "p85_tt_min": p85_tt.round(1),
            "min_tt_min": min_tt.round(1),
            "max_tt_min": max_tt.round(1),
        }
    )


def build_desire_lines_gdf(
    zones: gpd.GeoDataFrame,
    od_df: pd.DataFrame,
    zone_field: str,
    min_trips: int = 1,
    top_n: int | None = None,
    selected_origin: str | None = None,
    selected_destination: str | None = None,
) -> gpd.GeoDataFrame:
    if od_df.empty:
        return gpd.GeoDataFrame(
            columns=[
                "origin_quartiere",
                "destination_quartiere",
                "n_trips",
                "median_tt_min",
                "mean_tt_min",
                "p85_tt_min",
                "geometry",
            ],
            geometry="geometry",
            crs=4326,
        )

    z = zones[[zone_field, "geometry"]].copy()
    z[zone_field] = z[zone_field].astype(str).str.strip()

    z_proj = z.to_crs(3857)
    cent_proj = z_proj.geometry.centroid
    cent = gpd.GeoSeries(cent_proj, crs=3857).to_crs(4326)

    centroids = pd.DataFrame(
        {
            zone_field: z[zone_field].tolist(),
            "cx": cent.x.tolist(),
            "cy": cent.y.tolist(),
        }
    )

    df = od_df.copy()
    df = df[df["n_trips"] >= min_trips].copy()
    df = df[df["origin_quartiere"] != df["destination_quartiere"]].copy()

    if selected_origin:
        df = df[df["origin_quartiere"] == selected_origin].copy()
    if selected_destination:
        df = df[df["destination_quartiere"] == selected_destination].copy()

    df = df.merge(
        centroids.rename(columns={zone_field: "origin_quartiere", "cx": "ox", "cy": "oy"}),
        on="origin_quartiere",
        how="left",
    )
    df = df.merge(
        centroids.rename(columns={zone_field: "destination_quartiere", "cx": "dx", "cy": "dy"}),
        on="destination_quartiere",
        how="left",
    )

    df = df.dropna(subset=["ox", "oy", "dx", "dy"]).copy()

    if top_n is not None and len(df) > top_n:
        df = df.sort_values("n_trips", ascending=False).head(top_n).copy()

    df["geometry"] = [
        LineString([(ox, oy), (dx, dy)])
        for ox, oy, dx, dy in zip(df["ox"], df["oy"], df["dx"], df["dy"])
    ]

    return gpd.GeoDataFrame(
        df[
            [
                "origin_quartiere",
                "destination_quartiere",
                "n_trips",
                "median_tt_min",
                "mean_tt_min",
                "p85_tt_min",
                "geometry",
            ]
        ],
        geometry="geometry",
        crs=4326,
    )


def build_hour_filtered_desire_lines_gdf(
    zones: gpd.GeoDataFrame,
    od_df: pd.DataFrame,
    trips_q: pd.DataFrame,
    trip_hours_df: pd.DataFrame,
    zone_field: str,
    hour_min: int,
    hour_max: int,
    min_trips: int = 1,
    top_n: int | None = None,
    selected_origin: str | None = None,
    selected_destination: str | None = None,
) -> gpd.GeoDataFrame:
    if trips_q.empty or trip_hours_df.empty:
        return gpd.GeoDataFrame(
            columns=[
                "origin_quartiere",
                "destination_quartiere",
                "n_trips",
                "median_tt_min",
                "mean_tt_min",
                "p85_tt_min",
                "geometry",
            ],
            geometry="geometry",
            crs=4326,
        )

    th = trip_hours_df.copy()
    th = th[(th["start_hour"] >= hour_min) & (th["start_hour"] <= hour_max)].copy()

    trips_h = trips_q.merge(th[["trip_id"]], on="trip_id", how="inner")
    if trips_h.empty:
        return gpd.GeoDataFrame(
            columns=[
                "origin_quartiere",
                "destination_quartiere",
                "n_trips",
                "median_tt_min",
                "mean_tt_min",
                "p85_tt_min",
                "geometry",
            ],
            geometry="geometry",
            crs=4326,
        )

    agg = (
        trips_h.groupby(["origin_quartiere", "destination_quartiere"], as_index=False)
        .agg(n_trips=("trip_id", "nunique"))
    )

    if {"median_tt_min", "mean_tt_min", "p85_tt_min"}.issubset(od_df.columns):
        od_metrics = od_df[
            ["origin_quartiere", "destination_quartiere", "median_tt_min", "mean_tt_min", "p85_tt_min"]
        ].drop_duplicates()
        agg = agg.merge(
            od_metrics,
            on=["origin_quartiere", "destination_quartiere"],
            how="left",
        )
    else:
        agg["median_tt_min"] = np.nan
        agg["mean_tt_min"] = np.nan
        agg["p85_tt_min"] = np.nan

    return build_desire_lines_gdf(
        zones=zones,
        od_df=agg,
        zone_field=zone_field,
        min_trips=min_trips,
        top_n=top_n,
        selected_origin=selected_origin,
        selected_destination=selected_destination,
    )


def make_desire_lines_map(
    zones: gpd.GeoDataFrame,
    zone_field: str,
    desire_lines_gdf: gpd.GeoDataFrame,
    selected_origin: str | None = None,
    selected_destination: str | None = None,
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
            "fillOpacity": 0.06,
        },
        tooltip=folium.GeoJsonTooltip(fields=[zone_field], aliases=["Quartiere"], sticky=False),
    ).add_to(m)

    if selected_origin:
        add_zone_outline(
            m, zones, zone_field, selected_origin,
            color="#000000", weight=3, dash_array="8, 8",
            fill_color="#000000", fill_opacity=0.06,
        )

    if selected_destination:
        add_zone_outline(
            m, zones, zone_field, selected_destination,
            color="#cc0000", weight=3,
            fill_color="#cc0000", fill_opacity=0.06,
        )

    if not desire_lines_gdf.empty:
        vals = desire_lines_gdf["n_trips"].dropna()
        vmin = float(vals.min())
        vmax = float(vals.max())
        if vmin == vmax:
            vmax = vmin + 1e-6

        colormap = make_high_contrast_trip_colormap(vmin, vmax)
        colormap.caption = "Number of trips"

        def line_style(feature):
            val = feature["properties"].get("n_trips")
            if val is None:
                return {"color": "#666666", "weight": 3, "opacity": 0.55}

            w = 3 + 7 * ((val - vmin) / (vmax - vmin)) if vmax > vmin else 5
            return {
                "color": colormap(val),
                "weight": w,
                "opacity": 0.85,
            }

        draw_gdf = desire_lines_gdf[
            [
                "origin_quartiere",
                "destination_quartiere",
                "n_trips",
                "median_tt_min",
                "mean_tt_min",
                "p85_tt_min",
                "geometry",
            ]
        ].copy()

        folium.GeoJson(
            json.loads(draw_gdf.to_json()),
            name="desire_lines",
            style_function=line_style,
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    "origin_quartiere",
                    "destination_quartiere",
                    "n_trips",
                    "median_tt_min",
                    "mean_tt_min",
                    "p85_tt_min",
                ],
                aliases=[
                    "Origin",
                    "Destination",
                    "Trips",
                    "Median TT (min)",
                    "Mean TT (min)",
                    "P85 TT (min)",
                ],
                sticky=False,
                labels=True,
            ),
        ).add_to(m)

        colormap.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def render_desire_lines_view(
    zones,
    zone_field,
    choices,
    od_path: str,
    trips_with_quartieri_path: str,
    trip_points_path: str,
):
    with st.sidebar:
        demo_mode = st.toggle("Use demo OD data", value=False, key="desire_demo_mode")

    od = build_demo_od(zones, zone_field) if demo_mode else load_od(od_path)
    trips_q = load_trips_with_quartieri(trips_with_quartieri_path)

    trip_ids_all = tuple(trips_q["trip_id"].astype(str).drop_duplicates().tolist())
    trip_hours_df = load_trip_hours_for_pairs(str(trip_points_path), trip_ids_all)

    with st.sidebar:
        desire_filter_mode = st.radio(
            "Filter mode",
            ["All OD pairs", "From one origin", "To one destination", "One origin to one destination"],
            index=0,
            key="desire_filter_mode",
        )

        selected_origin = None
        selected_destination = None

        if desire_filter_mode in ["From one origin", "One origin to one destination"]:
            st.selectbox("Origin quartiere", choices, key="shared_origin")
            selected_origin = st.session_state["shared_origin"]

        if desire_filter_mode in ["To one destination", "One origin to one destination"]:
            st.selectbox("Destination quartiere", choices, key="shared_destination")
            selected_destination = st.session_state["shared_destination"]

        min_trips = st.slider("Minimum trips", 1, 500, 20, 1, key="desire_min_trips")
        top_n = st.slider("Top OD pairs to draw", 10, 1000, 100, 10, key="desire_top_n")

    hour_min, hour_max = st.session_state["shared_hour_range"]

    desire_lines_gdf = build_hour_filtered_desire_lines_gdf(
        zones=zones,
        od_df=od,
        trips_q=trips_q,
        trip_hours_df=trip_hours_df,
        zone_field=zone_field,
        hour_min=hour_min,
        hour_max=hour_max,
        min_trips=min_trips,
        top_n=top_n,
        selected_origin=selected_origin,
        selected_destination=selected_destination,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("OD pairs shown", len(desire_lines_gdf))
    c2.metric("Min trips", min_trips)
    c3.metric("Hour window", f"{hour_min:02d}-{hour_max:02d}")
    c4.metric("Origin filter", selected_origin if selected_origin else "None")
    c5.metric("Destination filter", selected_destination if selected_destination else "None")

    m = make_desire_lines_map(
        zones=zones,
        zone_field=zone_field,
        desire_lines_gdf=desire_lines_gdf,
        selected_origin=selected_origin,
        selected_destination=selected_destination,
    )
    desire_map_data = st_folium(m, width=None, height=760)

    clicked_zone = None
    if desire_map_data and desire_map_data.get("last_clicked"):
        clicked_zone = get_zone_from_click(
            zones,
            zone_field,
            desire_map_data["last_clicked"]["lat"],
            desire_map_data["last_clicked"]["lng"],
        )

    update_zone_from_click(
        clicked_zone,
        "_pending_shared_origin",
        "_pending_shared_destination",
        "desire_use_origin",
        "desire_use_destination",
    )

    with st.expander("Desire line table"):
        if desire_lines_gdf.empty:
            st.write("No OD pairs found.")
        else:
            st.dataframe(
                desire_lines_gdf[
                    [
                        "origin_quartiere",
                        "destination_quartiere",
                        "n_trips",
                        "median_tt_min",
                        "mean_tt_min",
                        "p85_tt_min",
                    ]
                ].sort_values("n_trips", ascending=False),
                use_container_width=True,
            )

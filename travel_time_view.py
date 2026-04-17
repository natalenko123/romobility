import json

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from branca.colormap import linear
from streamlit_folium import st_folium

from config import MAP_CENTER, MAP_ZOOM
from loaders import load_od
from map_utils import add_focus_css, add_zone_outline, get_zone_from_click
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


def prepare_od_subset(
    od: pd.DataFrame,
    zone_field: str,
    selected_zone: str,
    direction_mode: str,
    metric: str,
    min_trips: int,
) -> pd.DataFrame:
    if direction_mode == "To selected quartiere":
        sub = od.loc[od["destination_quartiere"] == selected_zone].copy()
        sub = sub.loc[sub["n_trips"] >= min_trips]
        sub = sub.rename(columns={"origin_quartiere": zone_field})
    else:
        sub = od.loc[od["origin_quartiere"] == selected_zone].copy()
        sub = sub.loc[sub["n_trips"] >= min_trips]
        sub = sub.rename(columns={"destination_quartiere": zone_field})

    sub["map_value"] = sub[metric]
    keep_cols = [
        zone_field,
        "map_value",
        "n_trips",
        "median_tt_min",
        "mean_tt_min",
        "p85_tt_min",
        "min_tt_min",
        "max_tt_min",
    ]
    return sub[keep_cols]


def build_joined_layer(
    zones: gpd.GeoDataFrame,
    od: pd.DataFrame,
    zone_field: str,
    selected_zone: str,
    direction_mode: str,
    metric: str,
    min_trips: int,
) -> gpd.GeoDataFrame:
    sub = prepare_od_subset(od, zone_field, selected_zone, direction_mode, metric, min_trips)
    zones2 = zones.copy()
    zones2[zone_field] = zones2[zone_field].astype(str).str.strip()
    return zones2.merge(sub, on=zone_field, how="left")


def style_function_factory(colormap):
    def style_function(feature):
        val = feature["properties"].get("map_value")
        if val is None:
            return {"fillColor": "#d9d9d9", "color": "#666666", "weight": 0.8, "fillOpacity": 0.35}
        return {"fillColor": colormap(val), "color": "#333333", "weight": 1.0, "fillOpacity": 0.72}
    return style_function


def make_od_map(joined: gpd.GeoDataFrame, zones: gpd.GeoDataFrame, zone_field: str, selected_zone: str, metric_label: str) -> folium.Map:
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles="cartodbpositron")
    add_focus_css(m)

    vals = joined["map_value"].dropna()
    if len(vals) > 0:
        vmin = float(vals.min())
        vmax = float(vals.max())
        if vmin == vmax:
            vmax = vmin + 1e-6
        colormap = linear.YlOrRd_09.scale(vmin, vmax)
    else:
        colormap = linear.YlOrRd_09.scale(0, 1)
    colormap.caption = metric_label

    map_gdf = joined[[zone_field, "geometry", "map_value", "n_trips", "median_tt_min", "p85_tt_min"]].copy()

    folium.GeoJson(
        json.loads(map_gdf.to_json()),
        name="zones",
        style_function=style_function_factory(colormap),
        tooltip=folium.GeoJsonTooltip(
            fields=[zone_field, "map_value", "n_trips", "median_tt_min", "p85_tt_min"],
            aliases=["Quartiere", metric_label, "Trips", "Median TT (min)", "P85 TT (min)"],
            localize=True,
            sticky=False,
            labels=True,
            style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;",
        ),
    ).add_to(m)

    add_zone_outline(
        m,
        zones,
        zone_field,
        selected_zone,
        color="#000000",
        weight=3,
        dash_array="8, 8",
        fill_color="#000000",
        fill_opacity=0.10,
    )
    colormap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def render_travel_time_view(zones, zone_field, choices, od_path: str):
    metric_labels = {
        "median_tt_min": "Median travel time (min)",
        "mean_tt_min": "Mean travel time (min)",
        "p85_tt_min": "P85 travel time (min)",
        "min_tt_min": "Minimum travel time (min)",
        "max_tt_min": "Maximum travel time (min)",
    }

    with st.sidebar:
        demo_mode = st.toggle("Use demo OD data", value=False, key="od_demo_mode")
        st.selectbox("Selected quartiere", choices, key="od_zone")
        direction_mode = st.radio("Map meaning", ["To selected quartiere", "From selected quartiere"], index=0, key="od_direction_mode")
        metric = st.selectbox("Travel time metric", list(metric_labels.keys()), index=0, key="od_metric")
        min_trips = st.slider("Minimum trips per OD pair", 1, 50, 5, 1, key="od_min_trips")

    od = build_demo_od(zones, zone_field) if demo_mode else load_od(od_path)

    selected_zone = st.session_state["od_zone"]
    joined = build_joined_layer(zones, od, zone_field, selected_zone, direction_mode, metric, min_trips)

    n_valid = int(joined.loc[joined["map_value"].notna(), zone_field].nunique())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected quartiere", selected_zone)
    c2.metric("Colored zones", n_valid)
    c3.metric("Direction", "Inbound" if direction_mode == "To selected quartiere" else "Outbound")
    c4.metric("Mode", "Demo" if demo_mode else "Real data")

    m = make_od_map(joined, zones, zone_field, selected_zone, metric_labels[metric])
    map_data = st_folium(m, width=None, height=760)

    clicked_zone = None
    if map_data and map_data.get("last_clicked"):
        clicked_zone = get_zone_from_click(zones, zone_field, map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])

    c1, c2 = st.columns([2, 1])
    with c1:
        st.write(f"Clicked zone: {clicked_zone if clicked_zone else 'None'}")
    with c2:
        if clicked_zone and st.button("Use clicked zone", key="od_use_clicked"):
            st.session_state["_pending_od_zone"] = clicked_zone
            st.rerun()

    with st.expander("Preview joined table"):
        preview = joined[[zone_field, "map_value", "n_trips", "median_tt_min", "p85_tt_min"]].sort_values("map_value", na_position="last").reset_index(drop=True)
        st.dataframe(preview, use_container_width=True)

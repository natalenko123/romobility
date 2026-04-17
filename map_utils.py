import json

import folium
import geopandas as gpd
from branca.colormap import LinearColormap
from shapely.geometry import Point


def add_focus_css(m: folium.Map):
    m.get_root().html.add_child(
        folium.Element(
            """
            <style>
            .leaflet-interactive:focus {
                outline: none !important;
            }
            </style>
            """
        )
    )


def get_zone_from_click(
    zones: gpd.GeoDataFrame,
    zone_field: str,
    lat: float,
    lon: float,
) -> str | None:
    pt = Point(lon, lat)

    hit = zones[zones.geometry.contains(pt)]
    if hit.empty:
        hit = zones[zones.geometry.intersects(pt)]
    if hit.empty:
        return None

    return str(hit.iloc[0][zone_field]).strip()


def prepare_zone_choices(zones: gpd.GeoDataFrame, zone_field: str) -> list[str]:
    return sorted(zones[zone_field].dropna().astype(str).unique().tolist())


def add_zone_outline(
    m: folium.Map,
    zones: gpd.GeoDataFrame,
    zone_field: str,
    selected_zone: str,
    color: str = "#000000",
    weight: int = 3,
    dash_array: str | None = None,
    fill_color: str | None = None,
    fill_opacity: float = 0.0,
):
    sel = zones.loc[
        zones[zone_field].astype(str).str.strip() == str(selected_zone).strip(),
        [zone_field, "geometry"],
    ]
    if sel.empty:
        return

    style = {"fillOpacity": fill_opacity, "color": color, "weight": weight, "interactive": False}
    if dash_array:
        style["dashArray"] = dash_array
    if fill_color:
        style["fillColor"] = fill_color

    folium.GeoJson(json.loads(sel.to_json()), style_function=lambda _: style, control=False).add_to(m)


def make_high_contrast_trip_colormap(vmin: float, vmax: float) -> LinearColormap:
    if vmin == vmax:
        vmax = vmin + 1e-6

    return LinearColormap(
        colors=["#c49a00", "#2ca25f", "#de2d26", "#000000"],
        index=[
            vmin,
            vmin + 0.33 * (vmax - vmin),
            vmin + 0.66 * (vmax - vmin),
            vmax,
        ],
        vmin=vmin,
        vmax=vmax,
    )

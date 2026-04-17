import io
import zipfile
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import streamlit as st
from google.transit import gtfs_realtime_pb2
from h3 import latlng_to_cell


@st.cache_data(show_spinner=False)
from pathlib import Path
import geopandas as gpd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_zones(path: str, zone_field: str, simplify_tolerance: float) -> gpd.GeoDataFrame:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Zones file not found: {p}")

    if p.suffix.lower() == ".shp":
        required = [
            p,
            p.with_suffix(".shx"),
            p.with_suffix(".dbf"),
        ]
        missing = [str(x) for x in required if not x.exists()]
        if missing:
            raise FileNotFoundError(f"Missing shapefile components: {missing}")

    gdf = gpd.read_file(p, engine="pyogrio")

    if zone_field not in gdf.columns:
        raise ValueError(f"Field '{zone_field}' not found. Available: {list(gdf.columns)}")
    if gdf.crs is None:
        raise ValueError("Zone file has no CRS. Assign one before running.")

    gdf = gdf[[zone_field, "geometry"]].copy().dropna(subset=["geometry"])
    gdf[zone_field] = gdf[zone_field].astype(str).str.strip()
    gdf = gdf.to_crs(4326)
    gdf = gdf.dissolve(by=zone_field, as_index=False)

    if simplify_tolerance > 0:
        gdf["geometry"] = gdf["geometry"].simplify(simplify_tolerance, preserve_topology=True)

    return gdf


@st.cache_data(show_spinner=False)
def load_od(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"OD file not found: {p}")

    df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)

    expected = {
        "origin_quartiere",
        "destination_quartiere",
        "n_trips",
        "median_tt_min",
        "mean_tt_min",
        "p85_tt_min",
        "min_tt_min",
        "max_tt_min",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"OD table missing columns: {sorted(missing)}")

    df = df.copy()
    df["origin_quartiere"] = df["origin_quartiere"].astype(str).str.strip()
    df["destination_quartiere"] = df["destination_quartiere"].astype(str).str.strip()

    for col in ["n_trips", "median_tt_min", "mean_tt_min", "p85_tt_min", "min_tt_min", "max_tt_min"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def load_trips_with_quartieri(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trips-with-quartieri file not found: {p}")

    df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)

    expected = {"trip_id", "origin_quartiere", "destination_quartiere"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Trips-with-quartieri table missing columns: {sorted(missing)}")

    df = df.copy()
    df["trip_id"] = df["trip_id"].astype(str)
    df["origin_quartiere"] = df["origin_quartiere"].astype(str).str.strip()
    df["destination_quartiere"] = df["destination_quartiere"].astype(str).str.strip()
    return df


@st.cache_data(show_spinner=False)
def load_trip_points_for_selected_ids(
    parquet_path: str,
    selected_trip_ids: tuple[str, ...],
    h3_resolution: int,
    hour_min: int,
    hour_max: int,
) -> pd.DataFrame:
    if not selected_trip_ids:
        return pd.DataFrame(columns=["trip_id", "ts", "latitude", "longitude", "h3_cell"])

    ids_df = pd.DataFrame({"trip_id": list(selected_trip_ids)})

    con = duckdb.connect()
    try:
        con.register("selected_ids", ids_df)
        query = f"""
            SELECT
                p.trip_id,
                p.ts,
                p.latitude,
                p.longitude
            FROM read_parquet('{parquet_path}') AS p
            INNER JOIN selected_ids AS s
                ON p.trip_id = s.trip_id
            WHERE p.latitude IS NOT NULL
              AND p.longitude IS NOT NULL
              AND p.ts IS NOT NULL
              AND EXTRACT(HOUR FROM p.ts) BETWEEN {hour_min} AND {hour_max}
            ORDER BY p.trip_id, p.ts
        """
        df = con.execute(query).df()
    finally:
        con.close()

    if df.empty:
        return pd.DataFrame(columns=["trip_id", "ts", "latitude", "longitude", "h3_cell"])

    df["trip_id"] = df["trip_id"].astype(str)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    df = df.dropna(subset=["trip_id", "ts", "latitude", "longitude"]).copy()
    df["h3_cell"] = [latlng_to_cell(lat, lon, h3_resolution) for lat, lon in zip(df["latitude"], df["longitude"])]
    return df


@st.cache_data(show_spinner=False)
def load_trip_hours_for_pairs(
    parquet_path: str,
    trip_ids: tuple[str, ...],
) -> pd.DataFrame:
    if not trip_ids:
        return pd.DataFrame(columns=["trip_id", "start_hour", "end_hour"])

    ids_df = pd.DataFrame({"trip_id": list(trip_ids)})

    con = duckdb.connect()
    try:
        con.register("selected_ids", ids_df)
        query = f"""
            SELECT
                p.trip_id,
                EXTRACT(HOUR FROM MIN(p.ts)) AS start_hour,
                EXTRACT(HOUR FROM MAX(p.ts)) AS end_hour
            FROM read_parquet('{parquet_path}') AS p
            INNER JOIN selected_ids AS s
                ON p.trip_id = s.trip_id
            WHERE p.ts IS NOT NULL
            GROUP BY p.trip_id
        """
        df = con.execute(query).df()
    finally:
        con.close()

    if df.empty:
        return pd.DataFrame(columns=["trip_id", "start_hour", "end_hour"])

    df["trip_id"] = df["trip_id"].astype(str)
    df["start_hour"] = pd.to_numeric(df["start_hour"], errors="coerce")
    df["end_hour"] = pd.to_numeric(df["end_hour"], errors="coerce")
    return df.dropna(subset=["trip_id", "start_hour", "end_hour"])


@st.cache_data(show_spinner=False)
def load_osm_roads_for_rome(osm_roads_path: str) -> gpd.GeoDataFrame:
    p = Path(osm_roads_path)
    if not p.exists():
        raise FileNotFoundError(f"OSM roads file not found: {p}")

    roads = gpd.read_parquet(p)
    if roads.crs is None:
        roads = roads.set_crs(4326)
    else:
        roads = roads.to_crs(4326)

    if "street_name" not in roads.columns:
        if "name" in roads.columns:
            roads["street_name"] = roads["name"].fillna("Unnamed road").astype(str)
        else:
            roads["street_name"] = "Unnamed road"

    return roads


def _read_csv_from_zip_bytes(zip_bytes: bytes, name: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = {Path(n).name.lower(): n for n in zf.namelist()}
        key = name.lower()
        if key not in names:
            raise FileNotFoundError(f"{name} not found in GTFS zip")
        with zf.open(names[key]) as f:
            return pd.read_csv(f)


def _read_csv_from_folder(folder: Path, name: str) -> pd.DataFrame:
    p = folder / name
    if not p.exists():
        raise FileNotFoundError(f"{name} not found in {folder}")
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_gtfs_static(static_path: str):
    if not static_path:
        return (
            pd.DataFrame(),
            gpd.GeoDataFrame(columns=["stop_id", "stop_name", "geometry"], geometry="geometry", crs=4326),
        )

    p = Path(static_path)
    if not p.exists():
        return (
            pd.DataFrame(),
            gpd.GeoDataFrame(columns=["stop_id", "stop_name", "geometry"], geometry="geometry", crs=4326),
        )

    try:
        if p.is_file() and p.suffix.lower() == ".zip":
            zip_bytes = p.read_bytes()
            routes = _read_csv_from_zip_bytes(zip_bytes, "routes.txt")
            stops = _read_csv_from_zip_bytes(zip_bytes, "stops.txt")
        elif p.is_dir():
            routes = _read_csv_from_folder(p, "routes.txt")
            stops = _read_csv_from_folder(p, "stops.txt")
        else:
            return (
                pd.DataFrame(),
                gpd.GeoDataFrame(columns=["stop_id", "stop_name", "geometry"], geometry="geometry", crs=4326),
            )
    except Exception:
        return (
            pd.DataFrame(),
            gpd.GeoDataFrame(columns=["stop_id", "stop_name", "geometry"], geometry="geometry", crs=4326),
        )

    if "route_id" not in routes.columns:
        routes = pd.DataFrame()
    else:
        keep = [c for c in ["route_id", "route_short_name", "route_long_name", "route_type"] if c in routes.columns]
        routes = routes[keep].copy()

    if not {"stop_id", "stop_lat", "stop_lon"}.issubset(stops.columns):
        stops_gdf = gpd.GeoDataFrame(columns=["stop_id", "stop_name", "geometry"], geometry="geometry", crs=4326)
    else:
        stops = stops.copy()
        if "stop_name" not in stops.columns:
            stops["stop_name"] = stops["stop_id"]
        stops = stops.dropna(subset=["stop_lat", "stop_lon"]).copy()
        stops_gdf = gpd.GeoDataFrame(
            stops[["stop_id", "stop_name"]].copy(),
            geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
            crs=4326,
        )

    return routes, stops_gdf


@st.cache_data(ttl=15, show_spinner=False)
def fetch_gtfs_rt_vehicle_positions(vehicle_url: str) -> pd.DataFrame:
    resp = requests.get(vehicle_url, timeout=20)
    resp.raise_for_status()

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(resp.content)

    rows = []
    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue

        v = entity.vehicle
        pos = v.position if v.HasField("position") else None
        trip = v.trip if v.HasField("trip") else None
        veh = v.vehicle if v.HasField("vehicle") else None

        if pos is None or not pos.HasField("latitude") or not pos.HasField("longitude"):
            continue

        rows.append(
            {
                "entity_id": entity.id if entity.id else None,
                "trip_id": trip.trip_id if trip and trip.trip_id else None,
                "route_id": trip.route_id if trip and trip.route_id else None,
                "start_date": trip.start_date if trip and trip.start_date else None,
                "direction_id": trip.direction_id if trip and trip.HasField("direction_id") else None,
                "vehicle_id": veh.id if veh and veh.id else None,
                "vehicle_label": veh.label if veh and veh.label else None,
                "latitude": float(pos.latitude),
                "longitude": float(pos.longitude),
                "bearing": float(pos.bearing) if pos.HasField("bearing") else np.nan,
                "speed_m_s": float(pos.speed) if pos.HasField("speed") else np.nan,
                "timestamp": int(v.timestamp) if v.HasField("timestamp") else np.nan,
                "current_stop_sequence": int(v.current_stop_sequence) if v.HasField("current_stop_sequence") else np.nan,
                "stop_id": v.stop_id if v.stop_id else None,
                "current_status": int(v.current_status) if v.HasField("current_status") else np.nan,
                
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    return df


def join_vehicle_positions_with_routes(vehicle_df: pd.DataFrame, routes_df: pd.DataFrame) -> pd.DataFrame:
    if vehicle_df.empty:
        return vehicle_df
    if routes_df.empty or "route_id" not in routes_df.columns:
        out = vehicle_df.copy()
        out["route_label"] = out["route_id"]
        return out

    out = vehicle_df.merge(routes_df, on="route_id", how="left")
    if "route_short_name" in out.columns:
        out["route_label"] = out["route_short_name"].fillna(out["route_id"])
    else:
        out["route_label"] = out["route_id"]
    return out

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
from shapely.geometry import LineString


@st.cache_data(show_spinner=False)
def load_zones(path: str, zone_field: str, simplify_tolerance: float) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)

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


def _is_url(path_or_url: str) -> bool:
    if not path_or_url:
        return False
    s = str(path_or_url).strip().lower()
    return s.startswith("http://") or s.startswith("https://")


@st.cache_data(show_spinner=False)
def _read_source_bytes(path_or_url: str) -> bytes:
    if _is_url(path_or_url):
        resp = requests.get(path_or_url, timeout=60)
        resp.raise_for_status()
        return resp.content

    p = Path(path_or_url)
    if not p.exists():
        raise FileNotFoundError(f"GTFS static source not found: {p}")
    return p.read_bytes()


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
def load_gtfs_table(static_path: str, name: str) -> pd.DataFrame:
    if not static_path:
        return pd.DataFrame()

    try:
        if _is_url(static_path):
            zip_bytes = _read_source_bytes(static_path)
            return _read_csv_from_zip_bytes(zip_bytes, name)

        p = Path(static_path)
        if p.is_file() and p.suffix.lower() == ".zip":
            zip_bytes = _read_source_bytes(static_path)
            return _read_csv_from_zip_bytes(zip_bytes, name)
        if p.is_dir():
            return _read_csv_from_folder(p, name)
    except Exception:
        return pd.DataFrame()

    return pd.DataFrame()


def _normalize_direction_id(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "none" or s.lower() == "nan":
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s


@st.cache_data(show_spinner=False)
def load_gtfs_static(static_path: str):
    routes = load_gtfs_table(static_path, "routes.txt")
    stops = load_gtfs_table(static_path, "stops.txt")

    if "route_id" in routes.columns:
        keep = [c for c in ["route_id", "route_short_name", "route_long_name", "route_type"] if c in routes.columns]
        routes = routes[keep].copy()
        routes["route_id"] = routes["route_id"].astype(str).str.strip()
        if "route_short_name" in routes.columns:
            routes["route_short_name"] = routes["route_short_name"].astype(str).str.strip()
        if "route_long_name" in routes.columns:
            routes["route_long_name"] = routes["route_long_name"].astype(str).str.strip()
    else:
        routes = pd.DataFrame()

    if not {"stop_id", "stop_lat", "stop_lon"}.issubset(stops.columns):
        stops_gdf = gpd.GeoDataFrame(columns=["stop_id", "stop_name", "geometry"], geometry="geometry", crs=4326)
    else:
        stops = stops.copy()
        stops["stop_id"] = stops["stop_id"].astype(str).str.strip()
        if "stop_name" not in stops.columns:
            stops["stop_name"] = stops["stop_id"]
        stops["stop_name"] = stops["stop_name"].astype(str).str.strip()
        stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
        stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
        stops = stops.dropna(subset=["stop_lat", "stop_lon"]).copy()

        stops_gdf = gpd.GeoDataFrame(
            stops[["stop_id", "stop_name"]].copy(),
            geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
            crs=4326,
        )

    return routes, stops_gdf


@st.cache_data(show_spinner=False)
def load_gtfs_route_map_data(
    static_path: str,
    route_id: str,
    direction_id: str | None = None,
):
    empty_lines = gpd.GeoDataFrame(
        columns=["route_id", "direction_id", "shape_id", "n_trips", "geometry"],
        geometry="geometry",
        crs=4326,
    )
    empty_stops = gpd.GeoDataFrame(
        columns=["route_id", "direction_id", "trip_id", "stop_id", "stop_name", "stop_sequence", "geometry"],
        geometry="geometry",
        crs=4326,
    )

    route_value = str(route_id).strip()
    direction_value = _normalize_direction_id(direction_id)

    trips = load_gtfs_table(static_path, "trips.txt")
    shapes = load_gtfs_table(static_path, "shapes.txt")
    stops = load_gtfs_table(static_path, "stops.txt")
    stop_times = load_gtfs_table(static_path, "stop_times.txt")

    if trips.empty or shapes.empty:
        return empty_lines, empty_stops

    required_trip_cols = {"trip_id", "route_id", "shape_id"}
    if not required_trip_cols.issubset(trips.columns):
        return empty_lines, empty_stops

    trips = trips.copy()
    trips["trip_id"] = trips["trip_id"].astype(str).str.strip()
    trips["route_id"] = trips["route_id"].astype(str).str.strip()
    trips["shape_id"] = trips["shape_id"].astype(str).str.strip()

    if "direction_id" in trips.columns:
        trips["direction_id"] = trips["direction_id"].apply(_normalize_direction_id)
    else:
        trips["direction_id"] = None

    trips = trips[(trips["route_id"] == route_value) & (trips["shape_id"] != "")]
    if direction_value is not None:
        trips = trips[trips["direction_id"] == direction_value]

    if trips.empty:
        return empty_lines, empty_stops

    shape_counts = (
        trips.groupby(["route_id", "direction_id", "shape_id"], dropna=False)
        .size()
        .reset_index(name="n_trips")
    )

    if direction_value is not None:
        selected_shapes = (
            shape_counts.sort_values(["n_trips", "shape_id"], ascending=[False, True])
            .head(1)
            .copy()
        )
    else:
        selected_shapes = (
            shape_counts.sort_values(["direction_id", "n_trips", "shape_id"], ascending=[True, False, True])
            .drop_duplicates(subset=["direction_id"])
            .copy()
        )
        if selected_shapes.empty:
            selected_shapes = (
                shape_counts.sort_values(["n_trips", "shape_id"], ascending=[False, True])
                .head(1)
                .copy()
            )

    if selected_shapes.empty:
        return empty_lines, empty_stops

    required_shape_cols = {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}
    if not required_shape_cols.issubset(shapes.columns):
        return empty_lines, empty_stops

    shapes = shapes.copy()
    shapes["shape_id"] = shapes["shape_id"].astype(str).str.strip()
    shapes["shape_pt_lat"] = pd.to_numeric(shapes["shape_pt_lat"], errors="coerce")
    shapes["shape_pt_lon"] = pd.to_numeric(shapes["shape_pt_lon"], errors="coerce")
    shapes["shape_pt_sequence"] = pd.to_numeric(shapes["shape_pt_sequence"], errors="coerce")
    shapes = shapes.dropna(subset=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]).copy()

    lines = []
    for shape_id, grp in shapes.groupby("shape_id"):
        grp = grp.sort_values("shape_pt_sequence")
        coords = list(zip(grp["shape_pt_lon"], grp["shape_pt_lat"]))
        if len(coords) >= 2:
            lines.append({"shape_id": shape_id, "geometry": LineString(coords)})

    if not lines:
        return empty_lines, empty_stops

    shape_lines = gpd.GeoDataFrame(lines, geometry="geometry", crs=4326)

    route_lines = selected_shapes.merge(shape_lines, on="shape_id", how="left")
    route_lines = route_lines.dropna(subset=["geometry"]).copy()
    route_lines_gdf = gpd.GeoDataFrame(route_lines, geometry="geometry", crs=4326)

    if stop_times.empty or stops.empty:
        return route_lines_gdf, empty_stops

    required_stop_time_cols = {"trip_id", "stop_id", "stop_sequence"}
    required_stop_cols = {"stop_id", "stop_lat", "stop_lon"}
    if not required_stop_time_cols.issubset(stop_times.columns) or not required_stop_cols.issubset(stops.columns):
        return route_lines_gdf, empty_stops

    rep_trips = (
        trips.merge(selected_shapes[["direction_id", "shape_id"]], on=["direction_id", "shape_id"], how="inner")
        .sort_values(["direction_id", "trip_id"], ascending=[True, True])
        .drop_duplicates(subset=["direction_id", "shape_id"])
        .copy()
    )

    if rep_trips.empty:
        return route_lines_gdf, empty_stops

    stop_times = stop_times.copy()
    stop_times["trip_id"] = stop_times["trip_id"].astype(str).str.strip()
    stop_times["stop_id"] = stop_times["stop_id"].astype(str).str.strip()
    stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"], errors="coerce")
    stop_times = stop_times.dropna(subset=["trip_id", "stop_id", "stop_sequence"]).copy()

    stops = stops.copy()
    stops["stop_id"] = stops["stop_id"].astype(str).str.strip()
    if "stop_name" not in stops.columns:
        stops["stop_name"] = stops["stop_id"]
    stops["stop_name"] = stops["stop_name"].astype(str).str.strip()
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops = stops.dropna(subset=["stop_lat", "stop_lon"]).copy()

    rep_trip_ids = tuple(rep_trips["trip_id"].astype(str).tolist())

    rep_stops = stop_times[stop_times["trip_id"].isin(rep_trip_ids)].copy()
    rep_stops = rep_stops.merge(
        rep_trips[["trip_id", "route_id", "direction_id"]],
        on="trip_id",
        how="left",
    )
    rep_stops = rep_stops.merge(
        stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]],
        on="stop_id",
        how="left",
    )
    rep_stops = rep_stops.dropna(subset=["stop_lat", "stop_lon"]).copy()
    rep_stops = rep_stops.sort_values(["direction_id", "trip_id", "stop_sequence"])

    stops_gdf = gpd.GeoDataFrame(
        rep_stops[["route_id", "direction_id", "trip_id", "stop_id", "stop_name", "stop_sequence"]].copy(),
        geometry=gpd.points_from_xy(rep_stops["stop_lon"], rep_stops["stop_lat"]),
        crs=4326,
    )

    return route_lines_gdf, stops_gdf


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

    df["timestamp_dt"] = (
        pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
        .dt.tz_convert("Europe/Rome")
    )
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
@st.cache_data(ttl=3600, show_spinner=False)
def load_gtfs_stop_tables(static_path: str):
    """
    Reads stops.txt, stop_times.txt, trips.txt from:
    - remote GTFS zip URL
    - local GTFS zip
    - local GTFS folder
    """
    if not static_path:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try:
        if _is_url(static_path):
            zip_bytes = _read_source_bytes(static_path)
            stops_df = _read_csv_from_zip_bytes(zip_bytes, "stops.txt")
            stop_times_df = _read_csv_from_zip_bytes(zip_bytes, "stop_times.txt")
            trips_df = _read_csv_from_zip_bytes(zip_bytes, "trips.txt")
        else:
            p = Path(static_path)

            if not p.exists():
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            if p.is_file() and p.suffix.lower() == ".zip":
                zip_bytes = _read_source_bytes(static_path)
                stops_df = _read_csv_from_zip_bytes(zip_bytes, "stops.txt")
                stop_times_df = _read_csv_from_zip_bytes(zip_bytes, "stop_times.txt")
                trips_df = _read_csv_from_zip_bytes(zip_bytes, "trips.txt")
            elif p.is_dir():
                stops_df = _read_csv_from_folder(p, "stops.txt")
                stop_times_df = _read_csv_from_folder(p, "stop_times.txt")
                trips_df = _read_csv_from_folder(p, "trips.txt")
            else:
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if not stops_df.empty and "stop_id" in stops_df.columns:
        stops_df["stop_id"] = stops_df["stop_id"].astype(str).str.strip()

    if not stop_times_df.empty:
        if "trip_id" in stop_times_df.columns:
            stop_times_df["trip_id"] = stop_times_df["trip_id"].astype(str).str.strip()
        if "stop_id" in stop_times_df.columns:
            stop_times_df["stop_id"] = stop_times_df["stop_id"].astype(str).str.strip()
        if "stop_sequence" in stop_times_df.columns:
            stop_times_df["stop_sequence"] = pd.to_numeric(stop_times_df["stop_sequence"], errors="coerce")

    if not trips_df.empty:
        if "trip_id" in trips_df.columns:
            trips_df["trip_id"] = trips_df["trip_id"].astype(str).str.strip()
        if "route_id" in trips_df.columns:
            trips_df["route_id"] = trips_df["route_id"].astype(str).str.strip()
        if "direction_id" in trips_df.columns:
            trips_df["direction_id"] = (
                trips_df["direction_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            )
        else:
            trips_df["direction_id"] = "unknown"

    return stops_df, stop_times_df, trips_df
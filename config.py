from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ZONES_PATH = DATA_DIR / "zones" / "2024_zmu_rsm.shp"
OD_MATRIX_PATH = DATA_DIR / "od" / "rome_od_quartiere_matrix.parquet"
TRIPS_WITH_QUARTIERI_PATH = DATA_DIR / "od" / "rome_trip_od_quartiere.parquet"
TRIP_POINTS_PATH = DATA_DIR / "trips" / "day=1_trips_passing_rome.parquet"

GTFS_STATIC_PATH = "https://romamobilita.it/sites/default/files/rome_static_gtfs.zip"
OSM_ROADS_PATH = DATA_DIR / "roads" / "rome_drive_edges.parquet"
GTFS_HISTORY_PATH = DATA_DIR / "gtfs_history.parquet"

VEHICLE_URL = "https://romamobilita.it/sites/default/files/rome_rtgtfs_vehicle_positions_feed.pb"
TRIP_UPDATES_URL = "https://romamobilita.it/sites/default/files/rome_rtgtfs_trip_updates_feed.pb"
ALERTS_URL = "https://romamobilita.it/sites/default/files/rome_rtgtfs_service_alerts_feed.pb"

ZONE_FIELD = "quartiere"
MAP_CENTER = [41.9028, 12.4964]
MAP_ZOOM = 11

SIMPLIFY_TOLERANCE = 0.0005
DEFAULT_MAX_TRIPS_TO_DRAW = 200
DEFAULT_TRIPS_SELECTED_ON_MAP = 20
DEFAULT_MAX_TRIPS_FOR_CORRIDOR = 500
DEFAULT_H3_RESOLUTION = 9
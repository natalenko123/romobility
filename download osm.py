from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd

OUT_PATH = Path(r"F:\mobito\osm\rome_drive_edges.parquet")


def normalize_name(x):
    if isinstance(x, list):
        if len(x) == 0:
            return None
        return " | ".join(str(v).strip() for v in x if v is not None and str(v).strip() != "")
    if pd.isna(x) or x is None:
        return None
    s = str(x).strip()
    return s if s else None


def build_rome_roads():
    G = ox.graph_from_place("Lazio, Italy", network_type="drive", simplify=True)
    _, edges = ox.graph_to_gdfs(G)

    roads = edges.copy().reset_index()

    if "name" not in roads.columns:
        roads["name"] = None
    if "highway" not in roads.columns:
        roads["highway"] = None
    if "length" not in roads.columns:
        roads["length"] = np.nan

    roads = roads[["u", "v", "key", "name", "highway", "length", "geometry"]].copy().to_crs(4326)

    roads["name"] = roads["name"].apply(normalize_name)

    def normalize_street_name(x):
        if x is None or str(x).strip() == "":
            return "Unnamed road"
        return str(x).strip()

    roads["street_name"] = roads["name"].apply(normalize_street_name)

    # optional: also normalize highway if it sometimes contains lists
    roads["highway"] = roads["highway"].apply(
        lambda x: " | ".join(map(str, x)) if isinstance(x, list) else (None if pd.isna(x) else str(x))
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    roads.to_parquet(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    build_rome_roads()
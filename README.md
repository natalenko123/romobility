# Rome app modular base

This package contains a modular starting point for your Streamlit mobility app.

Included now:
- `app.py`
- `config.py`
- `state.py`
- `ui_helpers.py`
- `map_utils.py`
- `loaders.py`
- `gtfs_rt_view.py`

Currently implemented in the modular app:
- Live buses (GTFS-RT)

The other tabs are intentionally not migrated yet. Move them one by one from your legacy monolithic file to avoid creating new hidden dependency errors.

## Install

```bash
pip install streamlit streamlit-folium geopandas pandas duckdb osmnx h3 gtfs-realtime-bindings requests streamlit-autorefresh
```

## Run

```bash
streamlit run app.py
```

## Notes

- `streamlit-autorefresh` is required because `st.autorefresh` is not a built-in Streamlit API.
- If you want stop names and route labels, provide `GTFS_STATIC_PATH` as a zip or folder with `routes.txt` and `stops.txt`.



## Newly added
- `travel_time_view.py`
- Travel time map is now modularized and wired into `app.py`

## Newly added\n- `trips_view.py`\n- Trips tab is now modularized and wired into `app.py`\n
## Newly added
- `corridor_view.py`
- Corridor tab is now modularized and includes most-used streets in results tables


## Updated
- local `OSM_ROADS_PATH` added in `config.py`
- `loaders.py` now reads precomputed OSM roads from local parquet
- `corridor_view.py` uses local OSM roads instead of downloading from OSM each run

## Newly added
- `desire_lines_view.py`
- Desire lines tab is now modularized and wired into `app.py`


## Updated
- global `shared_hour_range` added
- Trips, Corridor, Desire lines, and Live buses now use the same shared hour filter
- Travel time map shows the shared filter but does not apply it because current OD input is aggregated

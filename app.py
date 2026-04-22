import streamlit as st

from config import (
    ZONES_PATH,
    OD_MATRIX_PATH,
    TRIPS_WITH_QUARTIERI_PATH,
    TRIP_POINTS_PATH,
    GTFS_STATIC_PATH,
    ZONE_FIELD,
    SIMPLIFY_TOLERANCE,
)
from corridor_view import render_corridor_view
from desire_lines_view import render_desire_lines_view
from gtfs_rt_view import render_gtfs_rt_view
from loaders import load_zones
from map_utils import prepare_zone_choices
from route_performance_view import render_route_performance_view
from segment_performance_view import render_segment_performance_view
from state import apply_pending_zone_updates, ensure_state_defaults
from travel_time_view import render_travel_time_view
from trips_view import render_trips_view


def main():
    st.set_page_config(page_title="Rome mobility app", layout="wide")
    st.title("Rome mobility app")

    apply_pending_zone_updates()

    with st.sidebar:
        st.header("Inputs")
        zones_path = st.text_input("Zones file", str(ZONES_PATH))
        od_path = st.text_input("OD matrix file", str(OD_MATRIX_PATH))
        trips_with_quartieri_path = st.text_input("Trips with quartieri file", str(TRIPS_WITH_QUARTIERI_PATH))
        trip_points_path = st.text_input("Trip points file", str(TRIP_POINTS_PATH))
        gtfs_static_path = st.text_input("Static GTFS zip/folder (optional)", str(GTFS_STATIC_PATH))
        zone_field = st.text_input("Zone field", ZONE_FIELD)

        app_mode = st.radio(
            "Mode",
            [
                "Travel time map",
                "Trips between two quartieri",
                "Corridor usage between two quartieri",
                "Desire lines between quartieri",
                "Live buses (GTFS-RT)",
                "Route performance",
                "Segment performance",
            ],
            index=4,
        )

        simplify_tolerance = st.number_input(
            "Geometry simplify tolerance",
            min_value=0.0,
            max_value=0.01,
            value=float(SIMPLIFY_TOLERANCE),
            step=0.0001,
            format="%.4f",
        )

    zones = load_zones(zones_path, zone_field, float(simplify_tolerance))
    choices = prepare_zone_choices(zones, zone_field)
    ensure_state_defaults(choices)

    if app_mode not in ["Live buses (GTFS-RT)", "Route performance", "Segment performance"]:
        st.slider("Hour range", 0, 23, (0, 23), 1, key="shared_hour_range")
        hour_min, hour_max = st.session_state["shared_hour_range"]

        if app_mode == "Travel time map":
            st.caption(
                f"Shared hour filter: {hour_min:02d}-{hour_max:02d}. "
                "This tab uses aggregated OD input, so the hour filter is not applied here."
            )

    if app_mode == "Travel time map":
        render_travel_time_view(
            zones=zones,
            zone_field=zone_field,
            choices=choices,
            od_path=od_path,
        )
    elif app_mode == "Trips between two quartieri":
        render_trips_view(
            zones=zones,
            zone_field=zone_field,
            choices=choices,
            trips_with_quartieri_path=trips_with_quartieri_path,
            trip_points_path=trip_points_path,
        )
    elif app_mode == "Corridor usage between two quartieri":
        render_corridor_view(
            zones=zones,
            zone_field=zone_field,
            choices=choices,
            trips_with_quartieri_path=trips_with_quartieri_path,
            trip_points_path=trip_points_path,
        )
    elif app_mode == "Desire lines between quartieri":
        render_desire_lines_view(
            zones=zones,
            zone_field=zone_field,
            choices=choices,
            od_path=od_path,
            trips_with_quartieri_path=trips_with_quartieri_path,
            trip_points_path=trip_points_path,
        )
    elif app_mode == "Live buses (GTFS-RT)":
        render_gtfs_rt_view(
            zones=zones,
            zone_field=zone_field,
            choices=choices,
            gtfs_static_path=gtfs_static_path,
        )
    elif app_mode == "Route performance":
        render_route_performance_view(
            gtfs_static_path=gtfs_static_path,
        )
    elif app_mode == "Segment performance":
        render_segment_performance_view(
            gtfs_static_path=gtfs_static_path,
        )


if __name__ == "__main__":
    main()
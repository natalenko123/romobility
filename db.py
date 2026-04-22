from __future__ import annotations

import pandas as pd
import streamlit as st
from supabase import Client, create_client


@st.cache_resource
def get_supabase() -> Client:
    if "SUPABASE_URL" not in st.secrets or "SUPABASE_KEY" not in st.secrets:
        raise RuntimeError("Missing SUPABASE_URL / SUPABASE_KEY in Streamlit secrets.")
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"],
    )


def _normalize_direction_id(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "none":
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s


def load_gtfs_last_24h() -> pd.DataFrame:
    supabase = get_supabase()

    cutoff = (pd.Timestamp.now(tz="Europe/Rome") - pd.Timedelta(hours=24)).isoformat()

    res = (
        supabase.table("gtfs_history")
        .select("*")
        .gte("snapshot_time", cutoff)
        .order("snapshot_time")
        .execute()
    )

    df = pd.DataFrame(res.data or [])
    if df.empty:
        return df

    df["snapshot_time"] = pd.to_datetime(
        df["snapshot_time"], errors="coerce", utc=True
    ).dt.tz_convert("Europe/Rome")

    numeric_cols = [
        "n_buses",
        "delayed",
        "pct_delayed",
        "avg_delay_min",
        "p95_delay_min",
        "min_delay_min",
        "max_delay_min",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("snapshot_time").reset_index(drop=True)


def load_gtfs_route_last_24h(route_id: str, direction_id: str | None = None) -> pd.DataFrame:
    supabase = get_supabase()

    route_value = str(route_id).strip()
    direction_value = _normalize_direction_id(direction_id)

    cutoff = (pd.Timestamp.now(tz="Europe/Rome") - pd.Timedelta(hours=24)).isoformat()

    query = (
        supabase.table("gtfs_route_history")
        .select("*")
        .eq("route_id", route_value)
        .gte("snapshot_time", cutoff)
        .order("snapshot_time")
    )

    if direction_value is not None and direction_value != "All":
        query = query.eq("direction_id", direction_value)

    res = query.execute()

    df = pd.DataFrame(res.data or [])
    if df.empty:
        return df

    df["snapshot_time"] = pd.to_datetime(
        df["snapshot_time"], errors="coerce", utc=True
    ).dt.tz_convert("Europe/Rome")

    if "direction_id" in df.columns:
        df["direction_id"] = (
            df["direction_id"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

    if "route_id" in df.columns:
        df["route_id"] = df["route_id"].astype(str).str.strip()

    numeric_cols = [
        "n_buses",
        "delayed",
        "pct_delayed",
        "avg_delay_min",
        "p95_delay_min",
        "min_delay_min",
        "max_delay_min",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("snapshot_time").reset_index(drop=True)


def load_gtfs_segment_events(
    route_id: str,
    direction_id: str | None = None,
    from_stop_id: str | None = None,
    to_stop_id: str | None = None,
    hours_back: int = 24,
) -> pd.DataFrame:
    supabase = get_supabase()

    route_value = str(route_id).strip()
    direction_value = _normalize_direction_id(direction_id)

    cutoff = (pd.Timestamp.now(tz="Europe/Rome") - pd.Timedelta(hours=hours_back)).isoformat()

    query = (
        supabase.table("gtfs_segment_events")
        .select("*")
        .eq("route_id", route_value)
        .gte("traversal_time", cutoff)
        .order("traversal_time")
    )

    if direction_value is not None and direction_value != "All":
        query = query.eq("direction_id", direction_value)

    if from_stop_id is not None:
        query = query.eq("from_stop_id", str(from_stop_id).strip())

    if to_stop_id is not None:
        query = query.eq("to_stop_id", str(to_stop_id).strip())

    res = query.execute()
    df = pd.DataFrame(res.data or [])
    if df.empty:
        return df

    df["traversal_time"] = pd.to_datetime(
        df["traversal_time"], errors="coerce", utc=True
    ).dt.tz_convert("Europe/Rome")

    time_cols = ["from_event_time", "to_event_time"]
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col], errors="coerce", utc=True
            ).dt.tz_convert("Europe/Rome")

    numeric_cols = ["tt_min", "from_stop_sequence", "to_stop_sequence"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "direction_id" in df.columns:
        df["direction_id"] = (
            df["direction_id"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

    if "route_id" in df.columns:
        df["route_id"] = df["route_id"].astype(str).str.strip()

    if "from_stop_id" in df.columns:
        df["from_stop_id"] = df["from_stop_id"].astype(str).str.strip()

    if "to_stop_id" in df.columns:
        df["to_stop_id"] = df["to_stop_id"].astype(str).str.strip()

    return df.sort_values("traversal_time").reset_index(drop=True)
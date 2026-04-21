from __future__ import annotations

import pandas as pd
import streamlit as st
from supabase import Client, create_client


@st.cache_resource
def get_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


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
        df["snapshot_time"], utc=True, errors="coerce"
    ).dt.tz_convert("Europe/Rome")

    return df
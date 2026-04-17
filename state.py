import streamlit as st


def apply_pending_zone_updates():
    mapping = {
        "_pending_od_zone": "od_zone",
        "_pending_shared_origin": "shared_origin",
        "_pending_shared_destination": "shared_destination",
    }
    for pending_key, target_key in mapping.items():
        if pending_key in st.session_state:
            st.session_state[target_key] = st.session_state[pending_key]
            del st.session_state[pending_key]


def ensure_state_defaults(choices: list[str]) -> None:
    second = choices[1] if len(choices) > 1 else choices[0]
    defaults = {
        "od_zone": choices[0],
        "shared_origin": choices[0],
        "shared_destination": second,
        "shared_hour_range": (0, 23),
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

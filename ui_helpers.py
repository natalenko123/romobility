import streamlit as st


def update_zone_from_click(
    clicked_zone: str | None,
    origin_pending_key: str,
    dest_pending_key: str,
    origin_button_key: str,
    dest_button_key: str,
):
    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        st.write(f"Clicked zone: {clicked_zone if clicked_zone else 'None'}")

    with c2:
        if clicked_zone and st.button("Use clicked zone as origin", key=origin_button_key):
            st.session_state[origin_pending_key] = clicked_zone
            st.rerun()

    with c3:
        if clicked_zone and st.button("Use clicked zone as destination", key=dest_button_key):
            st.session_state[dest_pending_key] = clicked_zone
            st.rerun()

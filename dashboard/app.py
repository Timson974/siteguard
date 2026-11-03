"""SiteGuard dashboard (Streamlit). Talks only to the REST API."""
import base64
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="SiteGuard", page_icon="🦺", layout="wide")


def api_get(path: str, **params):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"API not reachable at {API_URL}: {e}")
        st.stop()


st.title("🦺 SiteGuard — PPE violations")

with st.sidebar:
    model = api_get("/model")
    st.subheader("Model")
    st.write(f"**{model['model']}**")
    if not model["ppe_model"]:
        st.warning("Fallback COCO model loaded: it finds people but not hard hats or vests. "
                   "Train the PPE model in Colab and put best.pt into models/.")
    st.caption("Violation classes: " + ", ".join(model["violation_classes"]))
    include_synth = st.checkbox("Include synthetic demo data", value=False,
                                help="Rows created by simulate_cameras.py --synthetic. "
                                     "Not real model output — for testing the dashboard only.")
    if st.button("Refresh"):
        st.rerun()

tab_overview, tab_try = st.tabs(["Overview", "Try an image"])

with tab_overview:
    s = api_get("/stats/summary", include_synthetic=include_synth)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames analysed", s["frames"])
    c2.metric("People detected", s["persons"])
    c3.metric("Violations", s["violations"])
    c4.metric("Avg latency, ms", round(s["avg_latency_ms"], 1))

    if s["frames"] == 0:
        st.info("No data yet. Send images with scripts/simulate_cameras.py or use the 'Try an image' tab.")
    else:
        left, right = st.columns(2)
        cams = pd.DataFrame(api_get("/stats/by-camera", include_synthetic=include_synth))
        if not cams.empty:
            left.subheader("Violations by camera")
            left.plotly_chart(px.bar(cams, x="camera_id", y="violations",
                                     hover_data=["frames", "violations_per_frame"]),
                              use_container_width=True)
        hours = pd.DataFrame(api_get("/stats/by-hour", include_synthetic=include_synth))
        if not hours.empty:
            right.subheader("Violations by hour of day")
            right.plotly_chart(px.bar(hours, x="hour", y="violations"), use_container_width=True)

        days = pd.DataFrame(api_get("/stats/by-day", include_synthetic=include_synth))
        if not days.empty:
            st.subheader("Violations by day and type")
            st.plotly_chart(px.bar(days, x="day", y="violations", color="violation_type"),
                            use_container_width=True)

        st.subheader("Latest violations")
        st.dataframe(pd.DataFrame(api_get("/violations/recent", limit=50,
                                          include_synthetic=include_synth)),
                     use_container_width=True, hide_index=True)

with tab_try:
    st.write("Upload a construction-site photo. The API runs the model and returns what it found.")
    camera = st.selectbox("Camera", ["cam_01", "cam_02", "cam_03", "cam_04"])
    save = st.checkbox("Save result to the database", value=True)
    up = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
    if up is not None:
        with st.spinner("Running model..."):
            r = requests.post(f"{API_URL}/detect",
                              files={"file": (up.name, up.getvalue(), up.type)},
                              data={"camera_id": camera, "save": str(save).lower(), "annotate": "true"},
                              timeout=60)
        if r.ok:
            res = r.json()
            a, b = st.columns([3, 2])
            a.image(base64.b64decode(res["annotated_image_b64"]), use_container_width=True)
            b.metric("Violations", res["n_violations"])
            b.metric("People", res["n_persons"])
            b.metric("Latency, ms", res["latency_ms"])
            if res["detections"]:
                b.dataframe(pd.DataFrame(res["detections"])[["class_name", "confidence", "is_violation"]],
                            hide_index=True)
        else:
            st.error(r.text)

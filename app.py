import streamlit as st
import joblib
import pandas as pd
import numpy as np
import datetime
import random

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NEO Guardian - Planetary Defense UI",
    layout="wide"
    
)
# ---------------- BACKGROUND IMAGE ----------------
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1475274047050-1d0c0975c63e?q=80&w=2070&auto=format&fit=crop");
             background-attachment: fixed;
             background-size: cover;
         }}
         /* Make the text inputs and containers slightly transparent so they pop */
         [data-testid="stHeader"] {{
             background-color: rgba(0,0,0,0);
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

import os
import requests

API_URL = os.getenv("NEO_API_URL", "http://localhost:8000")

# ---------------- LOAD MODEL (FALLBACK / LOCAL) ----------------
@st.cache_resource
def load_local_model():
    try:
        return joblib.load("asteroid_guardian_v2.pkl")
    except Exception:
        try:
            return joblib.load("asteroid_guardian_v1.pkl")
        except Exception:
            return None

local_model = load_local_model()

def query_prediction_api(payload: dict):
    """
    Queries the decoupled FastAPI backend if available;
    falls back to local model inference if backend is offline.
    """
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=2.0)
        if resp.status_code == 200:
            return resp.json(), "API"
    except Exception:
        pass
    
    # Local fallback
    if local_model is None:
        return None, "OFFLINE"
    
    input_df = pd.DataFrame([{
        "Absolute Magnitude": payload["absolute_magnitude"],
        "Orbit Uncertainity": payload["orbit_uncertainty"],
        "Relative Velocity km per sec": payload["relative_velocity"],
        "Eccentricity": payload["eccentricity"],
        "Miss Dist.(Astronomical)": payload["miss_distance"],
        "Minimum Orbit Intersection": payload["minimum_orbit_intersection"]
    }])[local_model.feature_names_in_]
    
    pred = int(local_model.predict(input_df)[0])
    prob = float(local_model.predict_proba(input_df)[0][1])
    
    # IAU formula
    diam_km = (1329 / np.sqrt(0.15)) * (10 ** (-0.2 * payload["absolute_magnitude"]))
    diam_m = round(float(diam_km * 1000), 2)
    
    if diam_m < 25:
        threat_class = "Meteoroid (Atmospheric Burn-up Likely)"
    elif diam_m < 140:
        threat_class = "City-Level Threat (Local Impact)"
    elif diam_m < 1000:
        threat_class = "Potentially Hazardous (Regional Impact)"
    else:
        threat_class = "Planet-Killer (Extinction Event)"
        
    return {
        "is_hazardous": (pred == 1),
        "status": "HAZARDOUS OBJECT" if pred == 1 else "SECURE / NOMINAL",
        "confidence_score": round(prob if pred == 1 else (1.0 - prob), 4),
        "hazard_probability": round(prob, 4),
        "estimated_diameter_meters": diam_m,
        "threat_classification": threat_class,
        "model_version": "v2.0-local"
    }, "LOCAL"

model = local_model

# ---------------- SESSION STORAGE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- HEADER ----------------
st.title("NEO Guardian")
st.markdown("### AI-Powered Planetary Defense UI")
st.caption("Translating complex astronomical data into actionable safety insights.")

# ---------------- NAVIGATION ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    " Project Briefing",
    " Threat Scanner",
    " The Science",
    " Mission Archive"
])

# -----------------------------------------------------
# TAB 1 : PROJECT OVERVIEW 
# ----------------------------------------------------
with tab1:
    st.header("Protecting Our Planet with Data")
    
    col_intro, col_img = st.columns([2, 1])
    
    with col_intro:
        st.write("""
        Near-Earth Objects (NEOs) are rocks orbiting the sun that occasionally cross paths with Earth. 
        While most are harmless, a few pose a risk. **NEO Guardian** uses a **Random Forest Machine Learning model** to scan these objects and predict danger levels instantly.
        """)
        
        with st.expander(" How does the AI know if an asteroid is dangerous?"):
            st.write("""
            The AI was trained on thousands of NASA records. It doesn't just look at 'distance'; it looks at 
            how **Size**, **Speed**, and **Orbit Shape** interact. For example, a small rock moving very fast 
            can be more dangerous than a large rock moving slowly!
            """)

    st.subheader(" Monitoring Systems")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Physicals**\n\nEstimates the object's mass based on how much sunlight it reflects (Magnitude).")
    with c2:
        st.info("**Dynamics**\n\nCalculates the kinetic energy based on velocity and orbital eccentricity.")
    with c3:
        st.info("**Proximity**\n\nDetermines the 'Closest Approach' (MOID) to Earth's personal orbit.")

# ----------------------------------------------------
# TAB 2 : THREAT SCANNER (The "Dashboard")
# ----------------------------------------------------
with tab2:
    st.header("Tactical Observation Dashboard")
    # --- QUICK SCENARIO PRESETS (Real NASA Observations) ---
    preset = st.selectbox(
        "⚡ Quick-Load Orbital Telemetry Preset",
        [
            "Custom Input (Configure Sliders Below)",
            "⚠️ High-Hazard Close Call (e.g. PHA 2433953 - Low MOID, High Brightness)",
            "🛡️ Nominal Deep-Space NEO (e.g. 3334820 - High MOID, Safe Clearance)",
            "☄️ High-Velocity Grazing Object (Hypervelocity, Significant Eccentricity)"
        ]
    )

    # Preset values dictionary
    default_vals = {
        "mag": 22.0, "unc": 5, "vel": 15.0, "ecc": 0.50, "miss": 0.05, "moid": 0.01
    }
    if "High-Hazard" in preset:
        default_vals = {"mag": 20.8, "unc": 0, "vel": 12.4, "ecc": 0.20, "miss": 0.19, "moid": 0.0001}
    elif "Nominal" in preset:
        default_vals = {"mag": 26.5, "unc": 6, "vel": 18.6, "ecc": 0.31, "miss": 0.49, "moid": 0.48}
    elif "High-Velocity" in preset:
        default_vals = {"mag": 21.0, "unc": 2, "vel": 32.5, "ecc": 0.65, "miss": 0.03, "moid": 0.015}

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Appearance")
            mag = st.slider("Brightness/Absolute Magnitude (H)", 10.0, 30.0, float(default_vals["mag"]), 
                            help="Lower H = Brighter/Larger. Asteroids with H < 22 are usually large enough to survive the atmosphere.")
            uncertainty = st.slider("Data Quality (U)", 0, 9, int(default_vals["unc"]), 
                                    help="0 means we have a perfect track. 9 means the path is highly unpredictable.")

        with col2:
            st.markdown("####  Movement")
            velocity = st.number_input("Speed (km/s)", 0.0, 100.0, float(default_vals["vel"]), 
                                       help="Typical NEOs travel at 20 km/s. Higher speed = higher impact damage.")
            eccentricity = st.slider("Orbit Ovalness", 0.0, 0.95, float(default_vals["ecc"]), 
                                     help="0 is a circle. 0.9 is a stretched oval (comet-like).")

        with col3:
            st.markdown("####  Gap Distance")
            miss_dist = st.number_input("Miss Distance (AU)", 0.0, 5.0, float(default_vals["miss"]), 
                                        help="1 AU is the Earth-Sun distance. Anything < 0.05 AU is a 'Close Call'.")
            orbit_int = st.number_input("Path Intersection (MOID)", 0.0, 1.0, float(default_vals["moid"]), 
                                        help="The closest point between Earth's orbit and the NEO's orbit.")

    if st.button(" INITIATE SCAN", use_container_width=True, type="primary"):
        asteroid_id = f"NEO-{random.randint(1000,9999)}"
        
        payload = {
            "absolute_magnitude": float(mag),
            "orbit_uncertainty": int(uncertainty),
            "relative_velocity": float(velocity),
            "eccentricity": float(eccentricity),
            "miss_distance": float(miss_dist),
            "minimum_orbit_intersection": float(orbit_int)
        }

        result, engine_mode = query_prediction_api(payload)
        
        if result is None:
            st.error("System offline: Neither API backend nor local model could be reached.")
            st.stop()
            
        prediction = 1 if result["is_hazardous"] else 0
        probability = result["hazard_probability"]
        diameter_meters = result["estimated_diameter_meters"]
        threat_class = result["threat_classification"]

        st.divider()
        res_col1, res_col2 = st.columns([1, 1])

        # --- RESULTS: Professional Status Boxes ---
        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                    <div style="background-color:#4B1010; padding:20px; border-radius:10px; border: 2px solid #FF4B4B; text-align: center;">
                        <h2 style="color:#FF4B4B; margin:0;">🚨 HAZARDOUS OBJECT</h2>
                        <p style="color:white; margin:5px 0 0 0;">Priority Tracking: <b>REQUIRED</b></p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="background-color:#104B2A; padding:20px; border-radius:10px; border: 2px solid #00CC96; text-align: center;">
                        <h2 style="color:#00CC96; margin:0;">✅ SECURE / NOMINAL</h2>
                        <p style="color:white; margin:5px 0 0 0;">Priority Tracking: <b>NOT REQUIRED</b></p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("AI Prediction Confidence", f"{result['confidence_score']*100:.1f}%")
            with m_col2:
                st.metric("Inference Engine", f"{engine_mode} ({result['model_version']})")

        # --- RESULTS: Technical Dimensions ---
        with res_col2:
            st.write("### 📐 Technical Dimensions")
            st.metric("Estimated Diameter", f"{diameter_meters:.2f} Meters")
            
            # Threat Classification Category
            if "Meteoroid" in threat_class:
                st.info(f" **Class:** {threat_class}")
            elif "City" in threat_class:
                st.warning(f" **Class:** {threat_class}")
            else:
                st.error(f" **Class:** {threat_class}")

        # --- VISUALIZATION: Bar Chart (Dynamically Scaled) ---
        st.divider()
        st.write("###  Comparison to Earth Landmarks")
        
        # Adaptive comparison set based on physical scale
        if diameter_meters < 50:
            comp_df = pd.DataFrame({
                "Object": ["School Bus (12m)", "Blue Whale (30m)", "Space Shuttle (56m)", "THIS NEO"],
                "Meters": [12, 30, 56, max(diameter_meters, 1.0)],
                "Type": ["Reference", "Reference", "Reference", "Detected NEO"]
            })
        elif diameter_meters < 300:
            comp_df = pd.DataFrame({
                "Object": ["Boeing 747 (70m)", "Statue of Liberty (93m)", "Great Pyramid (138m)", "Eiffel Tower (300m)", "THIS NEO"],
                "Meters": [70, 93, 138, 300, diameter_meters],
                "Type": ["Reference", "Reference", "Reference", "Reference", "Detected NEO"]
            })
        else:
            comp_df = pd.DataFrame({
                "Object": ["Eiffel Tower (300m)", "Empire State (443m)", "Burj Khalifa (828m)", "Mount Everest (8848m)", "THIS NEO"],
                "Meters": [300, 443, 828, 8848 if diameter_meters > 2000 else 828, diameter_meters],
                "Type": ["Reference", "Reference", "Reference", "Reference", "Detected NEO"]
            })
        
        st.bar_chart(comp_df, x="Object", y="Meters", color="Type")

        # --- VISUALIZATION: Orbital Geometry (True Orbital Overlap) ---
        st.write("### 🛰️ Orbital Path & Earth Orbit Proximity")
        st.info(f"""
        **Orbital Telemetry:**
        - **NEO Eccentricity ($e$):** {eccentricity:.2f} | **MOID Danger Gap:** {orbit_int:.4f} AU
        - Blue ring shows **Earth's circular orbit (1.0 AU)**.
        - Red dashed path visualizes the **NEO's elliptical orbit**.
        """)
        
        t = np.linspace(0, 2 * np.pi, 250)
        # Earth reference orbit
        earth_x = np.cos(t)
        earth_y = np.sin(t)
        
        # Semi-major axis estimation (scaled around 1 AU + eccentricity offset)
        a_neo = 1.0 + (eccentricity * 0.4)
        b_neo = a_neo * np.sqrt(max(0.01, 1 - eccentricity**2))
        neo_x = a_neo * np.cos(t) - (a_neo * eccentricity)  # shift focus to Sun at (0,0)
        neo_y = b_neo * np.sin(t)
        
        orbit_df = pd.DataFrame({
            "Earth Orbit X": earth_x,
            "Earth Orbit Y": earth_y,
            "NEO Path X": neo_x,
            "NEO Path Y": neo_y
        })
        st.line_chart(orbit_df[["Earth Orbit Y", "NEO Path Y"]])
        st.caption(f"Visual representation of orbital eccentricity (e={eccentricity}). Center (0,0) represents a circular reference orbit.")

        # --- LOGGING ---
        st.session_state.history.append({
            "Time": datetime.datetime.now().strftime("%H:%M:%S"),
            "ID": asteroid_id, 
            "Mag (H)": mag, 
            "Speed (km/s)": velocity, 
            "MOID (AU)": orbit_int,
            "Diameter (m)": int(diameter_meters),
            "Result": "HAZARDOUS" if prediction == 1 else "SAFE", 
            "Conf": f"{probability*100:.1f}%"
        })

# -----------------------------------------------------
# TAB 3 : THE SCIENCE (Analytics)
# ----------------------------------------------------
with tab3:
    st.header(" Behind the Intelligence")
    st.write("How does the system weight different astronomical factors?")
    
    col_anal1, col_anal2 = st.columns(2)
    
    with col_anal1:
        st.markdown("#### Feature Importance")
        importances_dict = None
        # Try fetching live importances from API
        try:
            r = requests.get(f"{API_URL}/model-info", timeout=1.5)
            if r.status_code == 200:
                importances_dict = r.json().get("feature_importances_pct", {})
        except Exception:
            pass

        # Fallback to local model inspection if API is offline
        if not importances_dict and hasattr(model, "feature_importances_") and hasattr(model, "feature_names_in_"):
            importances_dict = {
                name: round(float(imp) * 100, 2)
                for name, imp in zip(model.feature_names_in_, model.feature_importances_)
            }

        if importances_dict:
            label_map = {
                "Absolute Magnitude": "Brightness / Size (H)",
                "Minimum Orbit Intersection": "Path Intersection (MOID)",
                "Orbit Uncertainity": "Data Uncertainty (U)",
                "Relative Velocity km per sec": "Relative Velocity",
                "Eccentricity": "Orbit Eccentricity",
                "Miss Dist.(Astronomical)": "Miss Distance"
            }
            importance_df = pd.DataFrame([
                {"Feature": label_map.get(k, k), "Importance (%)": v}
                for k, v in importances_dict.items()
            ]).sort_values(by="Importance (%)", ascending=False)
            st.bar_chart(importance_df.set_index("Feature"), color="#FF4B4B")
        else:
            st.warning("Feature importance unavailable: Model is not currently loaded.")

    with col_anal2:
        st.markdown("####  The Math: Calculating Diameter")
        st.write("We use the Absolute Magnitude ($H$) to estimate the physical size ($D$) in kilometers:")
        st.latex(r"D = \frac{1329}{\sqrt{Albedo}} \cdot 10^{-0.2H}")
        st.info("The system assumes an Albedo (reflectivity) of 0.15, typical for rocky Near-Earth Asteroids.")
        
        st.markdown("####  Why Magnitude Matters?")
        st.write("Magnitude is a reverse scale. Lower numbers mean brighter (and usually larger) objects. A shift from H=22 to H=21 represents a significant increase in mass.")

# -----------------------------------------------------
# TAB 4 : MISSION ARCHIVE (Log)
# ----------------------------------------------------
with tab4:
    st.header(" Planetary Defense Log")
    st.write("Review and export the history of objects identified during this session.")

    if st.session_state.history:
        log_df = pd.DataFrame(st.session_state.history)
        
        # Custom coloring for the Status column
        def highlight_status(val):
            color = '#FF4B4B' if val == "HAZARDOUS" else '#00CC96'
            return f'color: {color}; font-weight: bold'

        st.dataframe(log_df.style.map(highlight_status, subset=['Result']), use_container_width=True, hide_index=True)
        
        st.divider()
        col_dl, col_clr = st.columns([1, 1])
        
        with col_dl:
            csv = log_df.to_csv(index=False).encode()
            st.download_button(" Export Mission Report (CSV)", csv, "neo_mission_log.csv", "text/csv", use_container_width=True)
        
        with col_clr:
            if st.button("🗑️ Purge Archive", use_container_width=True):
                st.session_state.history = []
                st.rerun()
    else:
        st.info("No active missions logged. Scan an object in the 'Threat Scanner' tab to begin recording data.")

# ---------------- FOOTER ----------------
st.divider()
st.write("Developed by Yasassri Ekanayake | My First EDA Data Science Project | © 2026")
st.write("Contact the creator at: yasas0721@gmail.com")

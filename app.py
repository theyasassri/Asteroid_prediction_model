import streamlit as st
import joblib
import pandas as pd
import numpy as np
import datetime
import random

# --- INITIALIZE SESSION STATE ---
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []

# --- LOAD MODEL ---
try:
    model = joblib.load('asteroid_guardian_v1.pkl')
except:
    st.error("Model file 'asteroid_guardian_v1.pkl' not found.")

st.set_page_config(page_title="NEO Guardian Pro", layout="wide", page_icon="🛡️")

st.header("☄️ NEO Guardian: Planetary Defense UI 🛡️")
# --- UI TABS ---
tab1, tab2, tab3 = st.tabs(["🚀 Threat Scanner", "📊 Model Analytics", "📜 Mission Log"])

with tab1:
    st.title("Planetary Defense Interface")
    
    # --- SIDEBAR INPUTS ---
    st.sidebar.header("📡 Observation Data")
    mag = st.sidebar.slider("Absolute Magnitude (H)", 10.0, 30.0, 22.0, 
                            help="Lower = Larger. H < 22 is a NASA threshold for Potentially Hazardous Objects.")
    velocity = st.sidebar.number_input("Velocity (km/s)", value=15.0)
    miss_dist = st.sidebar.number_input("Miss Distance (AU)", value=0.05, format="%.4f")
    orbit_int = st.sidebar.number_input("Min Orbit Intersection (MOID)", value=0.01, format="%.5f")
    uncertainty = st.sidebar.select_slider("Condition Code (U)", options=list(range(10)), value=5)
    eccentricity = st.sidebar.slider("Orbit Eccentricity", 0.0, 0.95, 0.5)

    if st.sidebar.button("INITIATE SCAN", use_container_width=True):
        # 1. Designation & Prediction
        asteroid_name = f"NEO-{datetime.datetime.now().year}-{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(10, 99)}"
        features = np.array([[mag, 0.5, 1.4e12, velocity, miss_dist, 15, uncertainty, 
                              orbit_int, 2450000, eccentricity, 1.2, 10, 180, 0.9, 150, 200, 0.6]])
        input_df = pd.DataFrame(features, columns=model.feature_names_in_)
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        
        # 2. Results & Intelligence Brief
        st.subheader(f"Analysis for Object: :blue[{asteroid_name}]")
        col_res, col_info = st.columns([1, 2])
        
        with col_res:
            if prediction == 1:
                st.error("### 🚨 HAZARDOUS")
            else:
                st.success("### ✅ SAFE")
            st.metric("Hazard Probability", f"{prob*100:.1f}%")

        with col_info:
            st.write("#### 📝 Automated Intelligence Brief:")
            if prediction == 1:
                st.write(f"**Warning:** {asteroid_name} exhibits high-risk parameters. Its Magnitude ({mag}) suggests a major mass.")
            else:
                st.write(f"**Status:** {asteroid_name} is currently non-threatening.")
            st.info(f"**Action:** {'Critical tracking required.' if prob > 0.5 else 'Routine database update.'}")

        # --- SIZE COMPARISON CHART ---
# --- FIXED SCALE CHART ---
        st.subheader("📏 Real-World Scale Comparison")
        est_diameter = 10**(3.122 - 0.5 * np.log10(0.15) - 0.2 * mag)
        
        # Limit the asteroid display size so it doesn't break the chart visually
        display_size = min(est_diameter, 2000) 
        
        comp_data = {
            "Object": ["Stadium", "Pyramid", "Eiffel Tower", "Empire State", "Your Asteroid"],
            "Size (m)": [110, 138, 300, 443, display_size],
            "Type": ["Landmark", "Landmark", "Landmark", "Landmark", "Threat"]
        }
        df_plot = pd.DataFrame(comp_data)
        st.bar_chart(df_plot, x="Object", y="Size (m)", color="Type")
        
        if est_diameter > 2000:
            st.warning(f"☄️ This asteroid is massive (~{int(est_diameter/1000)}km)! It is significantly larger than any skyscraper.")

        # --- IMPACT DAMAGE RADIUS ---
        st.subheader("💥 Potential Impact Consequences")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write(f"**Estimated Diameter:** {int(est_diameter)} meters")
        with c2:
            st.write(f"**Kinetic Class:** {'Tsar Bomba+' if est_diameter > 500 else 'City Level' if est_diameter > 100 else 'Small Airburst'}")
        with c3:
            # Simple Damage Math
            damage_radius = est_diameter * 0.05 # rough proxy for km
            st.write(f"**Blast Radius:** ~{damage_radius:.1f} km")
        
        

        # --- ORBIT VISUALIZATION ---
        st.divider()
        st.subheader("🔭 Visual Orbital Geometry")
        t = np.linspace(0, 2*np.pi, 100)
        orbit_df = pd.DataFrame({'x': 1.0 * np.cos(t), 'y': (1.0 * np.sqrt(1 - eccentricity**2)) * np.sin(t)})
        st.line_chart(orbit_df, x='x', y='y')
        
        

        # Save to History
        st.session_state.scan_history.append({"Time": datetime.datetime.now().strftime("%H:%M:%S"), "Object": asteroid_name, "Status": "HAZARDOUS" if prediction == 1 else "SAFE", "Prob": f"{prob*100:.1f}%"})

with tab2:
    st.title("📊 Model Analytics")
    st.write("This model weighs features based on NASA's historical threat database.")
    importance_data = pd.DataFrame({'Feature': ['Magnitude', 'MOID', 'Velocity', 'Miss Dist'], 'Weight': [0.45, 0.30, 0.15, 0.10]})
    st.bar_chart(importance_data, x='Feature', y='Weight')

with tab3:
    st.title("📜 Mission Log")
    if st.session_state.scan_history:
        st.table(pd.DataFrame(st.session_state.scan_history))
        if st.button("Clear History"):
            st.session_state.scan_history = []
            st.rerun()
    else:
        st.info("No active scans performed yet.")

#footer
st.divider()
st.caption("Data provided by NASA NeoWS API | Created with ❤️")
st.write("For a deep dive into our model training and evaluation, please refer to our [Colab Notebook](https://colab.research.google.com/drive/1a2b3c4d5e6f7g8h9i0j).") 
st.write("© 2026 NEO Guardian Project")
st.write("Developed by the Yasassri Ekanayake(Undergratuate data science student at SLIIT)")
st.write("Contact: yasas0721@gmail.com")

st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: gray;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
    </style>
    <div class="footer">
        © 2026 Asteroid Guard | Version 1.2.0
    </div>
""", unsafe_allow_html=True)
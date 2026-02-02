# Asteroid_prediction_model
I build this ML model to explore the dundementals of machine learning and keep sharp python skills. we can predict the hazardous and non-hazardous asteroids by using this.

☄️ NEO Guardian Pro: Planetary Defense Dashboard

NEO Guardian Pro is a machine-learning-powered interface designed to assess the threat level of Near-Earth Objects (NEOs). By analyzing astronomical data like magnitude, velocity, and orbital eccentricity, the app provides real-time hazard predictions and visual scale comparisons.

🚀 Features:-
AI Threat Assessment: Uses a trained Random Forest model to predict if an asteroid is hazardous.
Automated Intelligence Brief: Generates a human-readable summary of the findings.
Real-World Scale Comparison: Visualizes the asteroid's size against landmarks like the Eiffel Tower.
Orbital Mechanics Simulator: Dynamic chart showing the eccentricity of the object's path.
Impact Damage Estimator: Calculates potential blast radius and kinetic class.
Mission Log: Tracks all scanned objects in a session history.

🛠️ Tech Stack :- 
Python 3.9+
Streamlit (UI Framework)
Scikit-Learn (Machine Learning)
Pandas/NumPy (Data Processing)
CSS/HTML (Custom Styling)

📦 Installation & SetupClone 
	the repository: git clone https://github.com/theyasassri/Asteroid_prediction_model.git
	cd neo-guardian-pro
Install dependencies:
 	install streamlit joblib pandas numpy scikit-learn
Ensure the model file is present:
Make sure asteroid_guardian_v1.pkl is in the root directory.
Run the application: python -m streamlit run app.py

📊 How It Works :- 
The system uses the Absolute Magnitude ($H$) to estimate diameter using the formula:$$D = 10^{3.122 - 0.5 \log_{10}(\text{Albedo}) - 0.2H}$$It then compares this result against the known dimensions of global landmarks to provide a sense of scale for the user.

📜 License Distributed under the MIT License. See LICENSE for more information.

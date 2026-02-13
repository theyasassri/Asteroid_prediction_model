# Asteroid_prediction_model
I build this ML model to explore the dundementals of machine learning and keep sharp python skills. we can predict the hazardous and non-hazardous asteroids by using this.

NEO Guardian:
AI-Powered Planetary Defense NEO Guardian is a specialized machine learning dashboard designed to analyze and predict the threat levels of Near-Earth Objects (NEOs). By leveraging NASA-sourced astronomical data, the system identifies "Potentially Hazardous Asteroids" (PHAs) before they pose a risk to Earth.


**Project Overview :** 
This project was developed as a comprehensive Exploratory Data Analysis (EDA) and Machine Learning application. It translates complex orbital parameters into a user-friendly "Tactical Command" interface.


**Key Features :**
**Real-time Threat Prediction:** Uses a Random Forest Classifier to determine safety status with a probability confidence score.# **Scientific Diameter Estimation:** Implements the IAU diameter formula based on Absolute Magnitude (*H*) and Albedo.

**Dynamic Orbital Geometry :** Visualizes the "ovalness" (eccentricity) of an object's path.

**Comparative Scale :** Automatically compares detected NEOs to famous Earth landmarks like the Burj Khalifa and the Eiffel Tower.**Mission Archive :** A session-based log that tracks every scanned object for further review.


**Tech Stack** 
* Language: Python 3.9+ 
* Frontend: Streamlit (Custom CSS for Dark Mode/Space Theme)
* Machine Learning: Scikit-Learn
* Data Science Tools: Pandas, NumPy, Joblib
* Visualization: Streamlit Native Charts & Matplotlib

**The Science Behind the App**

The core intelligence of NEO Guardian relies on 6 primary astronomical inputs:

* Absolute Magnitude (H): Used to calculate physical size.
* Relative Velocity: Determines the kinetic energy potential.
* Eccentricity (*e*): Defines the shape of the orbital path.
* Miss Distance: The gap between Earth and the object's current approach.
* MOID: Minimum Orbit Intersection Distance (The "danger zone" where paths cross).
* Uncertainty (U): The reliability of the orbital data.
* Diameter Calculation Formula$$D = \frac{1329}{\sqrt{A}} \cdot 10^{-0.2H}$$Where *D* is diameter in km, *A* is assumed albedo (0.15), and *H* is absolute magnitude.

📦 Installation & SetupClone 
	
	the repository: git clone https://github.com/theyasassri/Asteroid_prediction_model.git
	cd neo-guardian

Install dependencies:
 	
	install streamlit joblib pandas numpy scikit-learn

Ensure the model file is present:
Make sure asteroid_guardian_v1.pkl is in the root directory.
Run the application: 
	
	python -m streamlit run app.py

 How It Works :- 
The system uses the Absolute Magnitude ($H$) to estimate diameter using the formula:$$D = 10^{3.122 - 0.5 \log_{10}(\text{Albedo}) - 0.2H}$$It then compares this result against the known dimensions of global landmarks to provide a sense of scale for the user.

Google colab link = https://colab.research.google.com/drive/1P6ZOIIuzZA3QY1E6kIcEBWPYzyY55EYP?usp=sharing

📜 License Distributed under the MIT License. See LICENSE for more information.

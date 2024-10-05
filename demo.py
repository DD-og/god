import streamlit as st
from joblib import load

# Load models
@st.cache_resource
def load_model():
    return load('asteroid_threat_detection_dt_model.joblib')

dt_model = load_model()

st.title('Asteroid Threat Detection')

# User input for asteroid characteristics
st.header('Enter Asteroid Characteristics')
relative_velocity = st.number_input('Relative velocity (km/h)', min_value=0.0, value=15000.0)
miss_distance = st.number_input('Miss distance (km)', min_value=0.0, value=100000000.0)
absolute_magnitude = st.number_input('Absolute magnitude', min_value=0.0, value=26.5)
estimated_diameter = st.number_input('Estimated average diameter (km)', min_value=0.0, value=0.05)

# Create asteroid data from user input
asteroid = [[relative_velocity, miss_distance, absolute_magnitude, estimated_diameter]]

# Make prediction when user clicks the button
if st.button('Predict Threat'):
    # Make predictions
    dt_prediction = dt_model.predict(asteroid)

    # Display asteroid characteristics
    st.header('Asteroid Characteristics:')
    st.write(f"Relative velocity: {asteroid[0][0]} km/h")
    st.write(f"Miss distance: {asteroid[0][1]} km")
    st.write(f"Absolute magnitude: {asteroid[0][2]}")
    st.write(f"Estimated average diameter: {asteroid[0][3]} km")

    # Display prediction
    st.header('Prediction:')
    st.write("Decision Tree prediction:", "Hazardous" if dt_prediction[0] else "Not Hazardous")
    
    # Visualize the prediction
    if dt_prediction[0]:
        st.error('⚠️ This asteroid is classified as HAZARDOUS!')
    else:
        st.success('✅ This asteroid is classified as NOT HAZARDOUS.')

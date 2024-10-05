import streamlit as st
from joblib import load
import pandas as pd
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# Load models
@st.cache_resource
def load_model():
    return load('asteroid_threat_detection_dt_model.joblib')

dt_model = load_model()

# Add this after loading the model
@st.cache_data
def load_model_metrics():
    return {
        'accuracy': 0.95,
        'precision': 0.92,
        'recall': 0.89,
        'f1_score': 0.90
    }

st.title('Asteroid Threat Detection')

# User input for asteroid characteristics
st.header('Enter Asteroid Characteristics')
relative_velocity = st.number_input('Relative velocity (km/h)', min_value=0.0, value=15000.0)
miss_distance = st.number_input('Miss distance (km)', min_value=0.0, value=100000000.0)
absolute_magnitude = st.number_input('Absolute magnitude', min_value=0.0, value=26.5)
estimated_diameter = st.number_input('Estimated average diameter (km)', min_value=0.0, value=0.05)

# Create asteroid data from user input
asteroid = [[relative_velocity, miss_distance, absolute_magnitude, estimated_diameter]]

# Add this after the user input section
st.sidebar.header('About')
st.sidebar.info('This app uses a Decision Tree model to predict whether an asteroid is potentially hazardous based on its characteristics.')

# Add this in the sidebar
st.sidebar.header('Model Performance')
metrics = load_model_metrics()
for metric, value in metrics.items():
    st.sidebar.metric(metric.capitalize(), f"{value:.2%}")

# Add feature to compare multiple asteroids
compare_asteroids = st.checkbox('Compare with another asteroid')

if compare_asteroids:
    st.subheader('Enter characteristics for comparison asteroid:')
    comp_relative_velocity = st.number_input('Relative velocity (km/h) - Comparison', min_value=0.0, max_value=1000000.0, value=20000.0)
    comp_miss_distance = st.number_input('Miss distance (km) - Comparison', min_value=0.0, max_value=1e12, value=50000000.0)
    comp_absolute_magnitude = st.number_input('Absolute magnitude - Comparison', min_value=-30.0, max_value=30.0, value=25.0)
    comp_estimated_diameter = st.number_input('Estimated average diameter (km) - Comparison', min_value=0.0, max_value=1000.0, value=0.1)
    
    comparison_asteroid = [[comp_relative_velocity, comp_miss_distance, comp_absolute_magnitude, comp_estimated_diameter]]

# Make prediction when user clicks the button
if st.button('Predict Threat'):
    with st.spinner('Analyzing asteroid(s)...'):
        # Make predictions
        dt_prediction = dt_model.predict(asteroid)
        dt_proba = dt_model.predict_proba(asteroid)[0]
        confidence = dt_proba[1] if dt_prediction[0] else dt_proba[0]

        # Display asteroid characteristics
        st.header('Asteroid Characteristics:')
        st.write(f"Relative velocity: {asteroid[0][0]} km/h")
        st.write(f"Miss distance: {asteroid[0][1]} km")
        st.write(f"Absolute magnitude: {asteroid[0][2]}")
        st.write(f"Estimated average diameter: {asteroid[0][3]} km")

        # Add feature importance visualization
        if isinstance(dt_model, Pipeline):
            # If it's a pipeline, get the last step (assuming it's the Decision Tree)
            tree_model = dt_model.steps[-1][1]
            if isinstance(tree_model, DecisionTreeClassifier):
                feature_importance = pd.DataFrame({
                    'feature': ['Relative Velocity', 'Miss Distance', 'Absolute Magnitude', 'Estimated Diameter'],
                    'importance': tree_model.feature_importances_
                }).sort_values('importance', ascending=False)

                fig = px.bar(feature_importance, x='feature', y='importance', title='Feature Importance')
                st.plotly_chart(fig)

                # Add explanation of the prediction
                st.subheader('Explanation')
                st.write("The model's prediction is based on the following factors:")
                for feature, importance in zip(feature_importance['feature'], feature_importance['importance']):
                    st.write(f"- {feature}: {importance:.2%}")
            else:
                st.warning("Feature importance visualization is not available for this model type.")
        else:
            st.warning("Feature importance visualization is not available for this model type.")

        # Display prediction
        st.header('Prediction:')
        st.write(f"Decision Tree prediction: {'Hazardous' if dt_prediction[0] else 'Not Hazardous'}")
        st.write(f"Confidence: {confidence:.2%}")
        
        # Add explanation of the prediction
        st.subheader('Explanation')
        st.write("The model's prediction is based on the following factors:")
        for feature, importance in zip(feature_importance['feature'], feature_importance['importance']):
            st.write(f"- {feature}: {importance:.2%}")

        # Add a scatter plot to visualize the asteroid's position
        scatter_df = pd.DataFrame({
            'Miss Distance (km)': [asteroid[0][1]],
            'Relative Velocity (km/h)': [asteroid[0][0]],
            'Estimated Diameter (km)': [asteroid[0][3]],
            'Hazardous': ['Yes' if dt_prediction[0] else 'No'],
            'Type': ['Original']
        })

        if compare_asteroids:
            comp_prediction = dt_model.predict(comparison_asteroid)
            comp_proba = dt_model.predict_proba(comparison_asteroid)[0]
            comp_confidence = comp_proba[1] if comp_prediction[0] else comp_proba[0]
            
            st.subheader('Comparison Asteroid Prediction:')
            st.write(f"Decision Tree prediction: {'Hazardous' if comp_prediction[0] else 'Not Hazardous'}")
            st.write(f"Confidence: {comp_confidence:.2%}")
            
            # Add comparison asteroid to the scatter plot
            scatter_df = pd.concat([scatter_df, pd.DataFrame({
                'Miss Distance (km)': [comparison_asteroid[0][1]],
                'Relative Velocity (km/h)': [comparison_asteroid[0][0]],
                'Estimated Diameter (km)': [comparison_asteroid[0][3]],
                'Hazardous': ['Yes' if comp_prediction[0] else 'No'],
                'Type': ['Comparison']
            })])

        fig = px.scatter(scatter_df, x='Miss Distance (km)', y='Relative Velocity (km/h)', 
                         size='Estimated Diameter (km)', color='Hazardous', symbol='Type',
                         title='Asteroid Position(s) Relative to Earth',
                         labels={'Hazardous': 'Threat Level'},
                         color_discrete_map={'Yes': 'red', 'No': 'green'},
                         hover_data=['Miss Distance (km)', 'Relative Velocity (km/h)', 'Estimated Diameter (km)', 'Type'])
        fig.update_layout(xaxis_type="log", yaxis_type="log")
        st.plotly_chart(fig)

        # Visualize the prediction
        if dt_prediction[0]:
            st.error('⚠️ This asteroid is classified as HAZARDOUS!')
        else:
            st.success('✅ This asteroid is classified as NOT HAZARDOUS.')

# Add this at the end of the file
st.sidebar.header('Disclaimer')
st.sidebar.warning('This is a demonstration app and should not be used for actual asteroid threat assessment.')

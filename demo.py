from joblib import load

# Load models
dt_model = load('asteroid_threat_detection_dt_model.joblib')

# Example of likely non-hazardous asteroid
non_hazardous_asteroid = [[15000, 100000000, 26.5, 0.05]]

# Make predictions
dt_prediction = dt_model.predict(non_hazardous_asteroid)

# Print asteroid characteristics
print("Asteroid characteristics:")
print(f"Relative velocity: {non_hazardous_asteroid[0][0]} km/h")
print(f"Miss distance: {non_hazardous_asteroid[0][1]} km")
print(f"Absolute magnitude: {non_hazardous_asteroid[0][2]}")
print(f"Estimated average diameter: {non_hazardous_asteroid[0][3]} km")

# Print predictions
print("Decision Tree prediction:", "Hazardous" if dt_prediction[0] else "Not Hazardous")
import sys
import joblib
import numpy as np
import cv2
import mediapipe as mp
import math
import json

# Load models
models = {}
measurements = ['Height(cm)', 'Arm Length(cm)', 'Leg Length(cm)', 'Shoulder Width(cm)', 'Shirt Length(cm)', 'Waist Circumference(cm)']
for measurement in measurements:
    models[measurement] = joblib.load(f"{measurement}_model.pkl")

# Initialize Mediapipe holistic model
mp_holistic = mp.solutions.holistic
target_size = (640, 480)

# Function to extract features
def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        image_resized = cv2.resize(image, target_size)
        with mp_holistic.Holistic(static_image_mode=True) as holistic:
            results = holistic.process(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                arm_length_euc = math.sqrt((landmarks[12].x - landmarks[16].x)**2 + (landmarks[12].y - landmarks[16].y)**2)
                leg_length_euc = math.sqrt((landmarks[24].x - landmarks[28].x)**2 + (landmarks[24].y - landmarks[28].y)**2)
                shoulder_width_euc = math.sqrt((landmarks[12].x - landmarks[11].x)**2 + (landmarks[12].y - landmarks[11].y)**2)
                shirt_length_euc = math.sqrt((landmarks[12].x - landmarks[24].x)**2 + (landmarks[12].y - landmarks[24].y)**2)
                nose_to_ankle_distance = math.sqrt((landmarks[0].x - landmarks[28].x)**2 + (landmarks[0].y - landmarks[28].y)**2)
                hip_distance = math.sqrt((landmarks[23].x - landmarks[24].x)**2 + (landmarks[23].y - landmarks[24].y)**2)
                waist_width_fraction = 0.8  # Adjust based on the expected proportions
                waist_width = hip_distance * waist_width_fraction
                waist_circumference = waist_width * math.pi
                return np.array([nose_to_ankle_distance, arm_length_euc, leg_length_euc, shoulder_width_euc, shirt_length_euc, waist_circumference]).reshape(1, -1)
    return None

# Main execution
if __name__ == "__main__":
    image_path = sys.argv[1]
    features = extract_features(image_path)
    if features is not None:
        predicted_measurements = {measurement: models[measurement].predict(features)[0] for measurement in measurements}
        print(json.dumps(predicted_measurements))
    else:
        print(json.dumps({"error": "No pose landmarks found"}))

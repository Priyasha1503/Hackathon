import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import google.generativeai as palm

# Configure PaLM with your API key
palm.configure(api_key="AIzaSyDttWImENH0IrmVIwkLPJukLlFgHPTJQE")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def detect_gesture(frame: np.ndarray):
    """
    Detect and annotate hand landmarks using MediaPipe.
    Return the annotated frame and a 'gesture' string.
    """
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    gesture = "No hand detected"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the original frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # TODO: Implement actual gesture logic
            gesture = "Open Palm (Placeholder)"  # Just a placeholder
    
    return frame, gesture

# ------------- Streamlit UI -------------
st.title("Gesture-Based HCI System (Streamlit Front End)")
st.write("Use your webcam to capture a hand gesture, then see how PaLM interprets it.")

# Streamlit's snapshot-style camera input
camera_image = st.camera_input("Take a snapshot of your gesture")

if camera_image is not None:
    # Convert the uploaded snapshot to a CV2 image (BGR)
    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Detect gestures & annotate frame
    annotated_frame, gesture = detect_gesture(frame)
    
    # Generate a response from PaLM
    response = palm.generate_text(prompt=f"Recognized gesture: {gesture}. Suggest an action.")

    # Display the annotated image + model's response
    st.image(annotated_frame, channels="BGR", caption=f"Gesture: {gesture}")
    st.subheader("PaLM Model Response")
    st.write(response.result)  # or response if you're using older versioz
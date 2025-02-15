import cv2
import mediapipe as mp
import google.generativeai as palm

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Palmstext-Bison-001 (Gemini)
palm.configure(api_key="AIzaSyDttWImENH0IrmVIwkLPJukLlFgHPTJQE")
model = palm.GenerativeModel(model_name="models/text-bison-001")

def get_gesture(landmarks):
    """Detect simple hand gestures based on landmarks."""
    if landmarks[8].y < landmarks[6].y:  # Index finger up
        return "Hello"
    elif landmarks[12].y < landmarks[10].y:  # Middle finger up
        return "Okay"
    else:
        return "Unknown"

# Start webcam feed
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            gesture = get_gesture(landmarks)
            cv2.putText(frame, f'Gesture: {gesture}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if gesture != "Unknown":
                response = model.generate_content(f"What does {gesture} mean?")
                print(response.text)

    cv2.imshow("Gesture-Based HCI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

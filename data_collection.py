import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define dataset file and gestures
DATA_FILE = "gestures.csv"
GESTURES = ["handOpen", "ThumbsUp", "peace", "fuck", "handClose", "rock"]  # Add your gestures

def capture_gesture(label):
    cap = cv2.VideoCapture(0)
    data = []
    print(f"Capturing data for {GESTURES[label]}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
                data.append([label] + keypoints)
        
        cv2.imshow("Capture Gesture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Save the collected data
    with open("gestures.csv", "a") as f:
        np.savetxt(f, data, delimiter=",")
    print(f"Data for {GESTURES[label]} saved!")

# Capture gestures by running capture_gesture(label) for each gesture
capture_gesture(3)  # Replace 0 with other gesture indices as needed


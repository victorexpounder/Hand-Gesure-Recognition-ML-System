import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("gesture_model.h5")
GESTURES = ["handOpen", "ThumbsUp", "peace", "fuck", "handClose", "rock"]  # Match your gesture labels

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
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
            
            # Predict gesture
            keypoints = np.array(keypoints).reshape(1, -1)
            prediction = model.predict(keypoints)
            class_id = np.argmax(prediction)
            confidence = prediction[0][class_id]
            
            if confidence > 0.7:  # Confidence threshold
                cv2.putText(frame, GESTURES[class_id], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()

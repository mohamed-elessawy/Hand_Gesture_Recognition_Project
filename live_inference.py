import cv2
import mediapipe as mp
import numpy as np
import pickle
import warnings

# Suppress sklearn feature name warnings
warnings.filterwarnings("ignore", category=UserWarning)

print("Loading SVM model from disk...")
with open("deploy_svm.pkl", "rb") as f:
    deploy_model = pickle.load(f)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
print("Initializing live video inference. Press 'q' to terminate.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 1. Extract Raw XYZ
                raw_coords = []
                for landmark in hand_landmarks.landmark:
                    raw_coords.extend([landmark.x, landmark.y, landmark.z])
                
                # 2. Shift Origin to Wrist (Landmark 0)
                wrist_x, wrist_y, wrist_z = raw_coords[0], raw_coords[1], raw_coords[2]
                shifted_coords = []
                for i in range(0, len(raw_coords), 3):
                    shifted_coords.extend([
                        raw_coords[i] - wrist_x,
                        raw_coords[i+1] - wrist_y,
                        raw_coords[i+2] - wrist_z
                    ])
                
                # 3. Calculate Distance to Middle Finger Tip (Landmark 12)
                # In our shifted_coords list, Landmark 12 starts at index 36 (12 * 3)
                mid_tip_x = shifted_coords[36]
                mid_tip_y = shifted_coords[37]
                
                # Using the Pythagorean theorem exactly like your training code
                distance = np.sqrt(mid_tip_x**2 + mid_tip_y**2) + 1e-6
                
                # 4. Final Scaling
                normalized_coords = [val / distance for val in shifted_coords]
                
                # 5. Inference
                input_data = np.array(normalized_coords).reshape(1, -1)
                prediction = deploy_model.predict(input_data)[0]
                                
                cv2.putText(frame, f"Gesture: {prediction}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imshow('Hand Gesture Recognition - Live Inference', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Guaranteed hardware release even if the script crashes
    cap.release()
    cv2.destroyAllWindows()
    
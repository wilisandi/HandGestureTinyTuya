import cv2
import win32com.client
import mediapipe as mp
import tinytuya
import time
import math
import json

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

d = tinytuya.OutletDevice('', '', '') #(deviceId, ipAddress, localKey) from tinytuya 
d.set_version(3.3)

# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mp.solutions.drawing_utils
def list_available_webcams(max_index=10):
    available_webcams = []
    
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_webcams.append(i)
            cap.release()
    
    return available_webcams
# Test the function
webcams = list_available_webcams()
print(f"Available webcams: {webcams}")
for i, webcam in enumerate(webcams):
    print(f"{i}: {webcam}")
# Open a video capture object (0 for the default camera)
cap = cv2.VideoCapture(1)

def detect_gesture(hand_landmarks):
    # Landmark indices for finger tips
    finger_tips = [8, 12, 16, 20]
    
    # Count the number of fingers up
    fingers_up = 0
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers_up += 1

    if fingers_up == 2:
        # d.set_value(21,"colour",True)
        color_data = json.loads("{\"h\":0,\"s\":1000,\"v\":1000}")
        # d.set_value(24, color_data,True)
        return "Lampu Merah"
    elif fingers_up == 1:
        d.set_value(20,True,True)
        d.set_value(21,"white",True)
        return "Lampu Nyala"
    elif fingers_up == 4:
        # d.set_value(21,"colour",True)
        color_data = json.loads("{\"h\":270,\"s\":1000,\"v\":1000}")
        # d.set_value(24, color_data,True)
        return "Lampu Ungu"
    elif fingers_up == 5:
        # d.set_value(21,"colour",True)
        color_data = json.loads("{\"h\":310,\"s\":1000,\"v\":1000}")
        # d.set_value(24, color_data,True)
        return "Lampu Pink"
    elif fingers_up == 3:
        # d.set_value(21,"colour",True)
        color_data = json.loads("{\"h\":135,\"s\":1000,\"v\":1000}")
        # d.set_value(24, color_data,True)
        return "Lampu Hijau"
    else:
        d.set_value(20,False,True)
        return "Lampu Mati"
    
def calculate_distance(landmark1, landmark2):
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

with mp_hands.Hands(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect hands
        results = hands.process(frame_rgb)
        
        # Check if hands are detected
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                handedness = results.multi_handedness[idx].classification[0].label
                
                gesture = detect_gesture(hand_landmarks)
                # Display gesture on the frame
                cv2.putText(frame, f'{handedness}: {gesture}', (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display the frame with hand landmarks
        cv2.imshow('Hand Recognition', frame)
        
        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
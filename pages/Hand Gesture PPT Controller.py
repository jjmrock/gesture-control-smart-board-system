import numpy as np
import cv2
import mediapipe as mp
import pyautogui
import streamlit as st
from PIL import Image

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    max_num_hands=1
)

# Utility functions
def get_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle

def get_distance(landmark_list):
    if len(landmark_list) < 2:
        return 0
    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    L = np.hypot(x2 - x1, y2 - y1)
    return np.interp(L, [0, 1], [0, 1000])

# Initialize session state

# Gesture detection functions
def is_left_click(landmark_list, thumb_index_dist):
    return (
        get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
        thumb_index_dist > 50
    )

def is_right_click(landmark_list, thumb_index_dist):
    return (
        get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and
        thumb_index_dist > 50
    )

def slide_show(landmark_list, thumb_index_dist):
    return (
        get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        thumb_index_dist > 100
    )

def stop_slide_show(landmark_list):
    return (
        get_angle(landmark_list[17], landmark_list[18], landmark_list[20]) > 100 and  # Pinky extended
        get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50

    )

def process_frame(frame):
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    landmark_list = []
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp.solutions.drawing_utils.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        for lm in hand_landmarks.landmark:
            landmark_list.append((lm.x, lm.y))

        if len(landmark_list) >= 21:
            thumb_index_dist = get_distance([landmark_list[4], landmark_list[5]])

            if is_left_click(landmark_list, thumb_index_dist):
                print("Left click - Previous slide")
                pyautogui.press("left", presses=1, interval=0.5)

            elif is_right_click(landmark_list, thumb_index_dist):
                print("Right click - Next slide")
                pyautogui.press("right", presses=1, interval=0.5)

            elif slide_show(landmark_list, thumb_index_dist):
                print("Starting slideshow...")
                pyautogui.press("f5", presses=1, interval=0.5)


            elif stop_slide_show(landmark_list):
                print("Stopping slideshow...")
                pyautogui.press("esc", presses=1, interval=0.5)


    return frame

def main():
    st.title("Hand Gesture Control App")
    st.markdown("""
    ## Control your computer with hand gestures!
    - **Left Click (Previous Slide)**: flick your index finger with thumb extended
    - **Right Click (Next Slide)**: flick your middle finger with thumb extended 
    - **Start Slideshow**: only thumb extended 
    - **Stop Slideshow**: Only pinky finger extended, all others closed
    """)

    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        processed_frame = process_frame(frame)
        FRAME_WINDOW.image(processed_frame[:, :, ::-1])  # Convert BGR to RGB

    cap.release()

if __name__ == "__main__":
    main()

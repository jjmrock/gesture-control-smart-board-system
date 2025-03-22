import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time
from PIL import Image
import streamlit as st
import google.generativeai as genai

model = genai.GenerativeModel("gemini-1.5-flash")
genai.configure(api_key="AIzaSyD2SdJNADszm-p5fDflilGUQnNLcH0FShE")

st.set_page_config(layout="wide")
st.image("han.png")

with st.container():
    col1 = st.container()  # Top container
    with col1:
        run = st.checkbox('Run', value=True)
        FRAME_WINDOW = st.image([])  # Placeholder for webcam

with st.container():
    col2 = st.container()  # Bottom container
    with col2:
        st.title("Answer")
        out_text_area = st.subheader("")

prompt = """
Solve this math problem. 
Please explain your steps first, then give the final answer.
If the problem involves multiple steps, break them down clearly.
Give answers line by line , not in one single sentence. 
If i make a circle then : A means find area , P means find perimeter.
if x variable then find it ,
if matrix [] then solve it 
"""

# Initialize video capture, hand detector, and canvas
cap = cv2.VideoCapture(0)
detector = HandDetector(staticMode=False,
                        maxHands=2,
                        modelComplexity=1,
                        detectionCon=0.8,
                        minTrackCon=0.8)

cap.set(3, 1280)
cap.set(4, 780)
canvas1 = None
canvas2 = None
drawing = False
circle_center = None
circle_radius = None
selected_circle = None
circle_history = []
dragging_circle = False
drawn_points = []
prev_pos = None
Output_text = ""

# Variables for erase gesture delay
erase_start_time = None  # To store the start time of the erase gesture
erase_delay = 1.0  # Delay duration in seconds

def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def draw_circles_on_canvas():
    global canvas2
    canvas2 = np.zeros_like(img)  # Clear the canvas
    for circle in circle_history:
        center, radius = circle
        cv2.circle(canvas2, center, int(radius), (255, 0, 255), 5)

def find_circle_at_point(point):
    """Find a circle in history that is close to the given point."""
    for circle in circle_history:
        center, radius = circle
        if calculate_distance(point, center) <= radius:
            return circle
    return None




while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Initialize canvases if they are None
    if canvas1 is None:
        canvas1 = np.zeros_like(img)
    if canvas2 is None:
        canvas2 = np.zeros_like(img)

    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        hand1 = hands[0]
        lmList1 = hand1['lmList']
        fingers1 = detector.fingersUp(hand1)
        tipOfIndexFinger = lmList1[8][0:2]

        if fingers1 == [0, 1, 0, 0, 0]:  # Drawing mode
            current_pos = lmList1[8][0:2]
            if prev_pos is None:
                prev_pos = current_pos
            drawn_points.append(current_pos)
            cv2.line(canvas1, current_pos, prev_pos, (255, 0, 255), 6)
            prev_pos = current_pos

        elif fingers1 == [1, 0, 0, 0, 0]:  # Erase gesture
            if erase_start_time is None:
                erase_start_time = time.time()  # Start timing when gesture is detected
            elif time.time() - erase_start_time >= erase_delay:
                # Perform erase if gesture has been held for the delay duration
                canvas1 = np.zeros_like(img)
                canvas2 = np.zeros_like(img)
                drawn_points.clear()
                circle_history.clear()

        elif fingers1 == [0, 0, 0, 0, 1]:  # Thumb up gesture
            pil_image = Image.fromarray(combined_canvas)
            response = model.generate_content([prompt, pil_image])

            out_text_area.text(response.text)

        else:
            erase_start_time = None  # Reset if gesture is not detected
            prev_pos = None  # Reset previous position

        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2['lmList']
            fingers2 = detector.fingersUp(hand2)
            tipOfIndexFinger2 = lmList2[8][0:2]
            length, info, img = detector.findDistance(tipOfIndexFinger, tipOfIndexFinger2, img, color=(255, 0, 0), scale=5)

            if fingers1 == [1, 1, 1, 1, 1] and fingers2 == [1, 1, 1, 1, 1]:
                if not drawing:
                    drawing = True
                    circle_center = ((tipOfIndexFinger[0] + tipOfIndexFinger2[0]) // 2,
                                     (tipOfIndexFinger[1] + tipOfIndexFinger2[1]) // 2)
                    circle_radius = length / 2
                else:
                    circle_radius = length / 2
            else:
                if drawing:
                    drawing = False
                    if circle_center and circle_radius:
                        circle_history.append((circle_center, circle_radius))
                        draw_circles_on_canvas()
                    circle_center = None
                    circle_radius = None

        if fingers1 == [0, 1, 1, 1, 0]:
            if not dragging_circle:
                dragging_circle = True
                selected_circle = find_circle_at_point(tipOfIndexFinger)
                if selected_circle:
                    circle_history.remove(selected_circle)
                    draw_circles_on_canvas()

        elif fingers1 == [1, 1, 1, 1, 1] and dragging_circle:
            dragging_circle = False
            if selected_circle:
                circle_history.append(selected_circle)
                draw_circles_on_canvas()
                selected_circle = None

        if dragging_circle and selected_circle:
            selected_center, radius = selected_circle
            new_center = tipOfIndexFinger
            selected_circle = (new_center, radius)

    if circle_center and circle_radius:
        cv2.circle(img, circle_center, int(circle_radius), (255, 0, 0), 2)

    if dragging_circle and selected_circle:
        selected_center, radius = selected_circle
        cv2.circle(img, selected_center, int(radius), (255, 0, 255), 2)




    combined_canvas = cv2.addWeighted(canvas2, 0.7, canvas1, 0.7, 2)
    combined_img = cv2.addWeighted(img, 0.8, combined_canvas, 0.7, 2)
    cv2.imshow("Image", combined_img)


    FRAME_WINDOW.image( combined_img, channels='BGR')

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import os
import urllib.request
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_FILE = 'hand_landmarker.task'

def ensure_model_exists():
    if not os.path.exists(MODEL_FILE):
        print(f"Downloading {MODEL_FILE}...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_FILE)
        print("Download complete!")

def air_paint_studio() -> None:
    """Draw with your hands

    Args:
        None

    Returns:
        None
    """
    ensure_model_exists()

    base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    canvas = None
    px, py = 0, 0
    
    # UI Color Palette Options (BGR Format)
    colors = [
        {"name": "BLUE", "bgr": (255, 0, 0), "x1": 20, "x2": 170},
        {"name": "GREEN", "bgr": (0, 255, 0), "x1": 190, "x2": 340},
        {"name": "RED", "bgr": (0, 0, 255), "x1": 360, "x2": 510},
        {"name": "YELLOW", "bgr": (0, 255, 255), "x1": 530, "x2": 680},
        {"name": "PURPLE", "bgr": (255, 0, 255), "x1": 700, "x2": 850},
        {"name": "ERASER", "bgr": (0, 0, 0), "x1": 870, "x2": 1020}
    ]

    current_color = (0, 0, 255)  # Default Red
    brush_thickness = 10
    frame_timestamp_ms = 0

    print("Air Paint Studio Active!")
    print(" - 2 Fingers Up (Index + Middle): SELECTION / COLOR PICKER")
    print(" - 1 Finger Up (Index Only): DRAWING MODE")
    print(" - Pinch Thumb + Index in Selection Mode: ADJUST BRUSH SIZE")
    print(" - Press 'c' to CLEAR | Press 'q' to QUIT")

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            if canvas is None:
                canvas = np.zeros((h, w, 3), dtype=np.uint8)

            # --- Draw Top UI Header ---
            # Header background
            cv2.rectangle(frame, (0, 0), (w, 80), (50, 50, 50), -1)

            for btn in colors:
                # Color box
                cv2.rectangle(frame, (btn["x1"], 10), (btn["x2"], 70), btn["bgr"], -1)
                
                # Active selection border
                border_color = (255, 255, 255) if current_color == btn["bgr"] else (100, 100, 100)
                border_thickness = 4 if current_color == btn["bgr"] else 2
                cv2.rectangle(frame, (btn["x1"], 10), (btn["x2"], 70), border_color, border_thickness)
                
                # Text label
                text_color = (0, 0, 0) if btn["name"] in ["GREEN", "YELLOW"] else (255, 255, 255)
                cv2.putText(frame, btn["name"], (btn["x1"] + 15, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            # --- MediaPipe Tracking ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            frame_timestamp_ms += 33
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if results.hand_landmarks:
                lms = results.hand_landmarks[0]

                # Key coordinates
                tx, ty = int(lms[4].x * w), int(lms[4].y * h)   # Thumb Tip (4)
                ix, iy = int(lms[8].x * w), int(lms[8].y * h)   # Index Tip (8)
                mx, my = int(lms[12].x * w), int(lms[12].y * h) # Middle Tip (12)

                index_up = lms[8].y < lms[6].y
                middle_up = lms[12].y < lms[10].y

                # Mode 1: Selection & UI Mode (Index + Middle BOTH UP)
                if index_up and middle_up:
                    px, py = 0, 0
                    cv2.rectangle(frame, (ix - 10, iy - 10), (ix + 10, iy + 10), (255, 255, 255), 2)

                    # Dynamic Brush Size Adjustment via Thumb-Index Distance
                    pinch_dist = math.hypot(ix - tx, iy - ty)
                    brush_thickness = int(np.interp(pinch_dist, [20, 200], [3, 50]))

                    # Check Top UI Palette Button Hits
                    if iy < 80:
                        for btn in colors:
                            if btn["x1"] < ix < btn["x2"]:
                                current_color = btn["bgr"]

                    # Display Status
                    cv2.putText(frame, f"SELECT / SIZE: {brush_thickness}px", (1050, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Mode 2: Drawing Mode (Index ONLY UP)
                elif index_up and not middle_up:
                    # Don't draw over the top UI header
                    if iy > 85:
                        draw_radius = brush_thickness // 2 if current_color != (0,0,0) else 15
                        cv2.circle(frame, (ix, iy), draw_radius, current_color, -1)

                        if px == 0 and py == 0:
                            px, py = ix, iy

                        # Apply stroke to persistence canvas
                        active_thickness = brush_thickness * 2 if current_color == (0,0,0) else brush_thickness
                        cv2.line(canvas, (px, py), (ix, iy), current_color, active_thickness)
                        px, py = ix, iy
                    else:
                        px, py = 0, 0
                else:
                    px, py = 0, 0

            # --- Canvas Layering ---
            canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask_inv = cv2.threshold(canvas_gray, 1, 255, cv2.THRESH_BINARY_INV)
            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            final_output = cv2.add(frame_bg, canvas)

            cv2.imshow("Air Paint Studio", final_output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                canvas = np.zeros((h, w, 3), dtype=np.uint8)

    cap.release()
    cv2.destroyAllWindows()


air_paint_studio()
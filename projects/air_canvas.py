import os
import urllib.request
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_FILE = 'hand_landmarker.task'

def ensure_model_exists() -> None:
    """Downloads the official MediaPipe hand landmarker model if missing."""
    if not os.path.exists(MODEL_FILE):
        print(f"Downloading {MODEL_FILE}...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_FILE)
        print("Download complete!")

def run_air_canvas() -> None:
    """Tracks your finger tips, detects if you are in drawing position (you can draw then) or detection position

    Args:
        None

    Returns:
        None
    """
    ensure_model_exists()

    # 1. Initialize Modern MediaPipe Tasks API
    base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    canvas = None
    px, py = 0, 0
    draw_color = (0, 0, 255)
    brush_thickness = 8
    frame_timestamp_ms = 0

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.flip(frame, 1)
            h, w, _ = frame.shape

            if canvas is None:
                canvas = np.zeros((h, w, 3), dtype=np.uint8)

            # Convert to RGB & create MediaPipe Image
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Process frame with timestamp
            frame_timestamp_ms += 33
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if results.hand_landmarks:
                # Get the first detected hand
                hand_lms = results.hand_landmarks[0]

                # Index tip (8) and Middle tip (12)
                ix, iy = int(hand_lms[8].x * w), int(hand_lms[8].y * h)
                mx, my = int(hand_lms[12].x * w), int(hand_lms[12].y * h)

                index_up = hand_lms[8].y < hand_lms[6].y
                middle_up = hand_lms[12].y < hand_lms[10].y

                if index_up and middle_up:
                    px, py = 0, 0
                    cv.circle(frame, (ix, iy), 12, (255, 255, 255), cv.FILLED)
                    cv.putText(frame, "SELECTION MODE", (10, 50),
                                cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                elif index_up and not middle_up:
                    cv.circle(frame, (ix, iy), 8, draw_color, cv.FILLED)
                    cv.putText(frame, "DRAWING MODE", (10, 50),
                                cv.FONT_HERSHEY_SIMPLEX, 0.8, draw_color, 2)

                    if px == 0 and py == 0:
                        px, py = ix, iy

                    cv.line(canvas, (px, py), (ix, iy), draw_color, brush_thickness)
                    px, py = ix, iy
                else:
                    px, py = 0, 0

            # Composite canvas onto video frame
            canvas_gray = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
            _, mask_inv = cv.threshold(canvas_gray, 20, 255, cv.THRESH_BINARY_INV)
            frame_bg = cv.bitwise_and(frame, frame, mask=mask_inv)
            final_output = cv.add(frame_bg, canvas)

            cv.imshow("Air Canvas (MediaPipe Tasks API)", final_output)

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                canvas = np.zeros((h, w, 3), dtype=np.uint8)

    cap.release()
    cv.destroyAllWindows()


run_air_canvas()
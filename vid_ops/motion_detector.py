import cv2
import numpy as np

def run_motion_detector(min_area: int) -> None:
    """Detects motion in the video stream

    Args: 
        min_area (int): minimum area of the object to be considered

    Returns:
        None
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream / webcam.")
        return

    
    background_frame = None

    print("Starting motion detector stream... Press 'q' to exit. Press 'r' to reset background.")

    while True:

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        output_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        if background_frame is None:
            background_frame = blurred
            continue

        frame_delta = cv2.absdiff(background_frame, blurred)

        _, thresh_delta = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)

        dilated_delta = cv2.dilate(thresh_delta, None, iterations=2)

        contours, _ = cv2.findContours(
            dilated_delta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = False

        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue

            motion_detected = True
            
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        status_text = "MOTION DETECTED!" if motion_detected else "System Clear"
        status_color = (0, 0, 255) if motion_detected else (0, 255, 0)
        cv2.putText(
            output_frame,
            f"Status: {status_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )

        cv2.imshow("Webcam Live Feed (Motion Detector)", output_frame)
        cv2.imshow("Thresholded Motion Mask (Delta)", dilated_delta)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            background_frame = blurred
            print("Background reference reset.")

    cap.release()
    cv2.destroyAllWindows()


run_motion_detector(min_area=1500.0)
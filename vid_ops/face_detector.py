import cv2
#import numpy as np

def detect_faces_and_eyes() -> None:
    """Detects your face

    Args: 
        None

    Returns:
        None
    """
    # 1. Load pre-trained Haar Cascade XML classifiers built into OpenCV
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    if face_cascade.empty() or eye_cascade.empty():
        print("Error: Could not load Haar Cascade XML files.")
        return

    # 2. Start Webcam Feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    print("Starting face detection stream... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to Grayscale (Haar features operate on single-channel intensity)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Equalize histogram to improve contrast under varying lighting
        gray = cv2.equalizeHist(gray)

        # 3. Detect Faces
        # scaleFactor=1.1: Reduces image size by 10% at each image pyramid scale
        # minNeighbors=5: Retains a bounding box only if 5 neighboring candidate boxes overlap
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # 4. Process Each Detected Face
        for (x, y, w, h) in faces:
            # Draw blue rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Define Region of Interest (ROI) specifically for the face area
            # (Eyes exist inside the face, so limiting search speeds up execution)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Detect eyes inside face ROI
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(15, 15)
            )

            for (ex, ey, ew, eh) in eyes:
                # Draw green rectangle around eyes
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # 5. Display Result
        cv2.imshow("Real-Time Face & Eye Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


detect_faces_and_eyes()
import cv2
import numpy as np

def run_ar_overlay(target_path: str, overlay_path: str) -> None:
    # 1. Load target marker and replacement overlay
    img_target = cv2.imread(target_path)
    img_overlay = cv2.imread(overlay_path)

    if img_target is None or img_overlay is None:
        print("Error: Could not load target or overlay image.")
        return

    # Resize overlay image to match the dimensions of the target marker
    ht, wt, _ = img_target.shape
    img_overlay = cv2.resize(img_overlay, (wt, ht))

    # 2. Initialize ORB Detector
    orb = cv2.ORB_create(nfeatures=1000)
    kp_target, des_target = orb.detectAndCompute(img_target, None)

    # 3. Initialize Brute-Force Matcher (using HAMMING distance for binary descriptors)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    print("AR Overlay Engine Active... Point webcam at target image.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Compute keypoints and descriptors for webcam frame
        kp_frame, des_frame = orb.detectAndCompute(frame, None)

        if des_frame is not None and len(des_frame) > 10:
            # Match descriptors using k-Nearest Neighbors (k=2)
            matches = bf.knnMatch(des_target, des_frame, k=2)

            # Apply Lowe's Ratio Test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # Ensure we have enough point correspondences to calculate Homography
            if len(good_matches) > 15:
                # Extract coordinates of matched keypoints
                src_pts = np.float32([kp_target[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Compute Homography matrix with RANSAC outlier rejection
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if H is not None:
                    hf, wf, _ = frame.shape

                    # Warp overlay image to match the perspective of the target in the video
                    warped_overlay = cv2.warpPerspective(img_overlay, H, (wf, hf))

                    # Create a mask for compositing the warped overlay onto the webcam frame
                    mask_target = np.ones((ht, wt), dtype=np.uint8) * 255
                    warped_mask = cv2.warpPerspective(mask_target, H, (wf, hf))
                    _, mask_inv = cv2.threshold(warped_mask, 1, 255, cv2.THRESH_BINARY_INV)

                    # Black out target region in webcam frame and add warped overlay
                    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                    frame = cv2.add(frame_bg, warped_overlay)

        cv2.imshow("Real-Time AR Overlay (ORB + Homography)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


run_ar_overlay("../img_ops/images/painting.png", "../img_ops/images/painting.png")
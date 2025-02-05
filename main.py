import cv2
import numpy as np


def detect_blue_cube(frame):
    """Detect blue regions in the frame"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def track_cube():
    """Main function to track blue cube using SIFT and sparse optical flow with arrows"""
    cap = cv2.VideoCapture(0)
    sift = cv2.SIFT_create()
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                     minEigThreshold=1e-3) # modification du threshold

    prev_gray = None
    prev_pts = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect blue region
        mask = detect_blue_cube(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            roi = frame[y:y + h, x:x + w]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Ensure gray_roi matches prev_gray size
            if prev_gray is not None and prev_gray.shape != gray_roi.shape:
                gray_roi = cv2.resize(gray_roi, (prev_gray.shape[1], prev_gray.shape[0]))

            # Detect keypoints
            kp = sift.detect(gray_roi, None)
            curr_pts = np.float32([kp.pt for kp in kp]).reshape(-1, 1, 2) if kp else None

            if prev_gray is not None and prev_pts is not None and curr_pts is not None and len(prev_pts) > 0 and len(
                    curr_pts) > 0:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray_roi, prev_pts, None, **lk_params)

##############################################################################################################################
                # # Replace your arrow drawing code with this: # Ok mais pas ouf
                # for i, (new, old) in enumerate(zip(next_pts, prev_pts)):
                #     if status[i]:
                #         a, b = new.ravel()
                #         c, d = old.ravel()
                #         cv2.arrowedLine(
                #             roi,
                #             (int(c), int(d)),
                #             (int(a), int(b)),
                #             (0, 255, 0),           # Green color
                #             3,                     # Line width (increased from 2)
                #             tipLength=0.5         # Arrow head length (increased from 0.3)
                #         )

                # for i, (new, old) in enumerate(zip(next_pts, prev_pts)): # OK tier
                #     if status[i]:
                #         a, b = new.ravel()
                #         c, d = old.ravel()
                        
                #         # Draw thick base arrow
                #         cv2.arrowedLine(
                #             roi,
                #             (int(c), int(d)),
                #             (int(a), int(b)),
                #             (0, 0, 0),           # Black background
                #             4,                   # Very thick base
                #             tipLength=0.5
                #         )
                        
                #         # Draw highlighted arrow on top
                #         cv2.arrowedLine(
                #             roi,
                #             (int(c), int(d)),
                #             (int(a), int(b)),
                #             (0, 255, 0),         # Green highlight
                #             2,                   # Thinner highlight
                #             tipLength=0.5
                #         )
##############################################################################################################################

                # def draw_direction_arrow(img, pt1, pt2):
                #     dx = pt2[0] - pt1[0]
                #     dy = pt2[1] - pt1[1]
                    
                #     # Draw main arrow
                #     cv2.arrowedLine(
                #         img,
                #         pt1,
                #         pt2,
                #         (0, 255, 0),
                #         3,
                #         tipLength=0.5
                #     )
                    
                #     # Add direction indicator circle
                #     mid_x = int((pt1[0] + pt2[0]) // 2)
                #     mid_y = int((pt1[1] + pt2[1]) // 2)
                #     cv2.circle(img, (mid_x, mid_y), 3, (255, 255, 0), -1)

                # # Usage in tracking loop:
                # for i, (new, old) in enumerate(zip(next_pts, prev_pts)):
                #     if status[i]:
                #         a, b = new.ravel()
                #         c, d = old.ravel()
                #         draw_direction_arrow(roi, (int(c), int(d)), (int(a), int(b)))

##############################################################################################################################

                # Draw movement arrows
                for i, (new, old) in enumerate(zip(next_pts, prev_pts)):
                    if status[i]:
                        a, b = new.ravel()
                        c, d = old.ravel()
                        cv2.arrowedLine(roi, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2, tipLength=0.3)

            # Store current frame and points for next iteration
            prev_gray = gray_roi.copy()
            prev_pts = curr_pts if curr_pts is not None else prev_pts
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Cube Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    track_cube()
import cv2
import numpy as np
import sys
import warnings
from scipy.ndimage import convolve  # Updated import

def HornSchunck(im1, im2, alpha=0.0001, N=10):  # reduce N from 50 to 10 to speed up computation
    HSFilter = np.array([[1/12, 1/6, 1/12],
                         [1/6,    0, 1/6],
                         [1/12, 1/6, 1/12]], float)

    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    # Initial velocities
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)

    [dx, dy, dt] = computeDerivatives(im1, im2)

    # Iterate to refine U, V
    for _ in range(N):
        Un = convolve(U, HSFilter)
        Vn = convolve(V, HSFilter)
        derivatives = (dx * Un + dy * Vn + dt) / (alpha**2 + dx**2 + dy**2)
        U = Un - dx * derivatives
        V = Vn - dy * derivatives

    return U, V

def computeDerivatives(im1, im2):
    kX = np.array([[-1, 1],
                   [-1, 1]]) * 0.25
    kY = np.array([[-1, -1],
                   [1, 1]]) * 0.25
    fx = convolve(im1, kX) + convolve(im2, kX)
    fy = convolve(im1, kY) + convolve(im2, kY)
    ft = convolve(im1, np.ones((2, 2)) * 0.25) + convolve(im2, -np.ones((2, 2)) * 0.25)
    return fx, fy, ft

def drawOpticalflow(img, U, V, step=15, threshold=500):  # Increase the step from 7 to 15 so fewer vectors are drawn
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = U[y, x], V[y, x]
    
    # Create lines from (x1, y1) to (x2, y2) for each flow vector
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)

    # Convert grayscale image to BGR to overlay lines
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw the red vectors (from start points to end points of the flow vectors)
    for (x1, y1), (x2, y2) in lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color in BGR
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)  # Green circle at the starting point
    
    # Create a dynamic outline by drawing the boundary of the motion in yellow
    contours = getMotionContours(U, V, threshold)
    for contour in contours:
        cv2.polylines(vis, [contour], isClosed=True, color=(0, 255, 255), thickness=2)  # Yellow outline
    
    return vis

def getMotionContours(U, V, threshold=2):
    """Generate contours of the moving objects based on flow vectors."""
    # Compute the magnitude of the flow vectors
    magnitude = np.sqrt(U**2 + V**2)
    
    # Mask to store moving regions
    mask = np.zeros(U.shape, dtype=np.uint8)
    
    # Mark areas with significant motion (magnitude greater than threshold)
    mask[magnitude > threshold] = 255
    
    # Find contours around the motion regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def main():
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        cap = cv2.VideoCapture(0)

        success, prev_frame = cap.read()
        if not success:
            print("Error reading video file.")
            cap.release()
            sys.exit()

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        while True:
            success, frame = cap.read()
            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Compute optical flow
            U, V = HornSchunck(prev_gray, gray)
            vis = drawOpticalflow(gray, U, V)

            # Display the result
            cv2.imshow("Horn-Schunck Optical Flow", vis)

            prev_gray = gray  # Update for the next frame

            # Quit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Usage: python HornSchunck.py <video_file>")
        sys.exit()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()

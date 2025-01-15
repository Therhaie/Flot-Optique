
import cv2
import numpy as np
import sys
import warnings
from scipy.ndimage import convolve  # Retain for Horn-Schunck algorithm computation

def HornSchunck(im1, im2, alpha=0.0001, N=10):
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

def drawOpticalflow(img, U, V, step=15):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = U[y, x], V[y, x]
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def main():
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        cap = cv2.VideoCapture(filename)

        success, prev_frame = cap.read()
        if not success:
            print("Error reading video file.")
            cap.release()
            sys.exit()

        # Use CUDA for grayscale conversion
        gpu_prev_frame = cv2.cuda_GpuMat()
        gpu_prev_gray = cv2.cuda_GpuMat()
        gpu_curr_frame = cv2.cuda_GpuMat()
        gpu_curr_gray = cv2.cuda_GpuMat()

        gpu_prev_frame.upload(prev_frame)
        gpu_prev_gray = cv2.cuda.cvtColor(gpu_prev_frame, cv2.COLOR_BGR2GRAY)

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Upload frame to GPU
            gpu_curr_frame.upload(frame)
            gpu_curr_gray = cv2.cuda.cvtColor(gpu_curr_frame, cv2.COLOR_BGR2GRAY)

            # Download frames to CPU for Horn-Schunck computation
            prev_gray = gpu_prev_gray.download()
            curr_gray = gpu_curr_gray.download()

            # Compute optical flow
            U, V = HornSchunck(prev_gray, curr_gray)
            vis = drawOpticalflow(curr_gray, U, V)

            # Display the result
            cv2.imshow("Horn-Schunck Optical Flow", vis)

            gpu_prev_gray = gpu_curr_gray.clone()  # Update for the next frame

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

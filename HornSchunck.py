# import cv2
# import numpy as np
# import sys
# import warnings
# from scipy.ndimage import convolve  # Updated import

# def HornSchunck(im1, im2, alpha=0.0001, N=10):  # reduce N from 50 to 10 to speed up computation
#     HSFilter = np.array([[1/12, 1/6, 1/12],
#                          [1/6,    0, 1/6],
#                          [1/12, 1/6, 1/12]], float)

#     im1 = im1.astype(np.float32)
#     im2 = im2.astype(np.float32)

#     # Initial velocities
#     U = np.zeros_like(im1)
#     V = np.zeros_like(im1)

#     [dx, dy, dt] = computeDerivatives(im1, im2)

#     # Iterate to refine U, V
#     for _ in range(N):
#         Un = convolve(U, HSFilter)
#         Vn = convolve(V, HSFilter)
#         derivatives = (dx * Un + dy * Vn + dt) / (alpha**2 + dx**2 + dy**2)
#         U = Un - dx * derivatives
#         V = Vn - dy * derivatives

#     return U, V

# def computeDerivatives(im1, im2):
#     kX = np.array([[-1, 1],
#                    [-1, 1]]) * 0.25
#     kY = np.array([[-1, -1],
#                    [1, 1]]) * 0.25
#     fx = convolve(im1, kX) + convolve(im2, kX)
#     fy = convolve(im1, kY) + convolve(im2, kY)
#     ft = convolve(im1, np.ones((2, 2)) * 0.25) + convolve(im2, -np.ones((2, 2)) * 0.25)
#     return fx, fy, ft

# def drawOpticalflow(img, U, V, step=15, threshold=500):  # Increase the step from 7 to 15 so fewer vectors are drawn
#     h, w = img.shape[:2]
#     y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
#     fx, fy = U[y, x], V[y, x]
    
#     # Create lines from (x1, y1) to (x2, y2) for each flow vector
#     lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
#     lines = np.int32(lines)

#     # Convert grayscale image to BGR to overlay lines
#     vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
#     # Draw the red vectors (from start points to end points of the flow vectors)
#     for (x1, y1), (x2, y2) in lines:
#         cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color in BGR
#         cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)  # Green circle at the starting point
    
#     # Create a dynamic outline by drawing the boundary of the motion in yellow
#     contours = getMotionContours(U, V, threshold)
#     for contour in contours:
#         cv2.polylines(vis, [contour], isClosed=True, color=(0, 255, 255), thickness=2)  # Yellow outline
    
#     return vis

# def getMotionContours(U, V, threshold=2):
#     """Generate contours of the moving objects based on flow vectors."""
#     # Compute the magnitude of the flow vectors
#     magnitude = np.sqrt(U**2 + V**2)
    
#     # Mask to store moving regions
#     mask = np.zeros(U.shape, dtype=np.uint8)
    
#     # Mark areas with significant motion (magnitude greater than threshold)
#     mask[magnitude > threshold] = 255
    
#     # Find contours around the motion regions
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours




# def main():
#     if True:
#     # if len(sys.argv) == 2:
#         # filename = sys.argv[1]
#         cap = cv2.VideoCapture(0)

#         success, prev_frame = cap.read()
#         if not success:
#             print("Error reading video file.")
#             cap.release()
#             sys.exit()

#         prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             gray = cv2.blur(gray, (5, 5))  # Apply mean filter to filter the noise


#             # Compute optical flow
#             U, V = HornSchunck(prev_gray, gray)
#             vis = drawOpticalflow(gray, U, V)

#             # Display the result
#             cv2.imshow("Horn-Schunck Optical Flow", vis)

#             prev_gray = gray  # Update for the next frame

#             # Quit on 'q' key
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#     else:
#         print("Usage: python HornSchunck.py <video_file>")
#         sys.exit()

# if __name__ == "__main__":
#     warnings.filterwarnings("ignore")
#     main()

#################### implementation of mean filter and morphological filter #####################

# import cv2
# import numpy as np
# import sys
# import warnings
# from scipy.ndimage import convolve

# def HornSchunck(im1, im2, alpha=0.0001, N=10):
#     HSFilter = np.array([[1/12, 1/6, 1/12],
#                          [1/6,    0, 1/6],
#                          [1/12, 1/6, 1/12]], float)

#     im1 = im1.astype(np.float32)
#     im2 = im2.astype(np.float32)

#     U = np.zeros_like(im1)
#     V = np.zeros_like(im1)

#     [dx, dy, dt] = computeDerivatives(im1, im2)

#     for _ in range(N):
#         Un = convolve(U, HSFilter)
#         Vn = convolve(V, HSFilter)
#         derivatives = (dx * Un + dy * Vn + dt) / (alpha**2 + dx**2 + dy**2)
#         U = Un - dx * derivatives
#         V = Vn - dy * derivatives

#     return U, V

# def computeDerivatives(im1, im2):
#     kX = np.array([[-1, 1],
#                    [-1, 1]]) * 0.25
#     kY = np.array([[-1, -1],
#                    [1, 1]]) * 0.25
#     fx = convolve(im1, kX) + convolve(im2, kX)
#     fy = convolve(im1, kY) + convolve(im2, kY)
#     ft = convolve(im1, np.ones((2, 2)) * 0.25) + convolve(im2, -np.ones((2, 2)) * 0.25)
#     return fx, fy, ft

# def drawOpticalflow(img, U, V, step=15, threshold=500):
#     h, w = img.shape[:2]
#     y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
#     fx, fy = U[y, x], V[y, x]

#     lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
#     lines = np.int32(lines)

#     vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

#     for (x1, y1), (x2, y2) in lines:
#         cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
#         cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

#     contours = getMotionContours(U, V, threshold)
#     for contour in contours:
#         cv2.polylines(vis, [contour], isClosed=True, color=(0, 255, 255), thickness=2)

#     return vis

# def getMotionContours(U, V, threshold=2):
#     magnitude = np.sqrt(U**2 + V**2)
#     mask = np.zeros(U.shape, dtype=np.uint8)
#     mask[magnitude > threshold] = 255

#     # Apply morphological operations to clean up the mask
#     kernel = np.ones((3, 3), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

# def main():
#     if True:
#         cap = cv2.VideoCapture(0)

#         success, prev_frame = cap.read()
#         if not success:
#             print("Error reading video file.")
#             cap.release()
#             sys.exit()

#         prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#         prev_gray = cv2.medianBlur(prev_gray, 5)  # Apply median filter

#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             gray = cv2.medianBlur(gray, 5)  # Apply median filter

#             U, V = HornSchunck(prev_gray, gray)
#             vis = drawOpticalflow(gray, U, V)

#             cv2.imshow("Horn-Schunck Optical Flow", vis)

#             prev_gray = gray

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#     else:
#         print("Usage: python HornSchunck.py <video_file>")
#         sys.exit()

# if __name__ == "__main__":
#     warnings.filterwarnings("ignore")
#     main()


########################## work with the hsv representation ########################

# import cv2
# import numpy as np
# import sys
# import warnings
# from scipy.ndimage import convolve

# def HornSchunck(im1, im2, alpha=0.0001, N=10):
#     HSFilter = np.array([[1/12, 1/6, 1/12],
#                          [1/6,    0, 1/6],
#                          [1/12, 1/6, 1/12]], float)

#     im1 = im1.astype(np.float32)
#     im2 = im2.astype(np.float32)

#     U = np.zeros_like(im1)
#     V = np.zeros_like(im1)

#     [dx, dy, dt] = computeDerivatives(im1, im2)

#     for _ in range(N):
#         Un = convolve(U, HSFilter)
#         Vn = convolve(V, HSFilter)
#         derivatives = (dx * Un + dy * Vn + dt) / (alpha**2 + dx**2 + dy**2)
#         U = Un - dx * derivatives
#         V = Vn - dy * derivatives

#     return U, V

# def computeDerivatives(im1, im2):
#     kX = np.array([[-1, 1],
#                    [-1, 1]]) * 0.25
#     kY = np.array([[-1, -1],
#                    [1, 1]]) * 0.25
#     fx = convolve(im1, kX) + convolve(im2, kX)
#     fy = convolve(im1, kY) + convolve(im2, kY)
#     ft = convolve(im1, np.ones((2, 2)) * 0.25) + convolve(im2, -np.ones((2, 2)) * 0.25)
#     return fx, fy, ft

# def drawOpticalflow(img, U, V, step=15, threshold=500):
#     h, w = img.shape[:2]
#     y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
#     fx, fy = U[y, x], V[y, x]

#     lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
#     lines = np.int32(lines)

#     vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

#     for (x1, y1), (x2, y2) in lines:
#         cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
#         cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

#     contours = getMotionContours(U, V, threshold)
#     for contour in contours:
#         cv2.polylines(vis, [contour], isClosed=True, color=(0, 255, 255), thickness=2)

#     return vis

# def getMotionContours(U, V, threshold=2):
#     magnitude = np.sqrt(U**2 + V**2)
#     mask = np.zeros(U.shape, dtype=np.uint8)
#     mask[magnitude > threshold] = 255

#     kernel = np.ones((3, 3), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

# def main():
#     if True:
#         cap = cv2.VideoCapture(0)

#         success, prev_frame = cap.read()
#         if not success:
#             print("Error reading video file.")
#             cap.release()
#             sys.exit()

#         prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
#         prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#         prev_gray = cv2.medianBlur(prev_gray, 5)  # Apply median filter

#         # Define the range for red color in HSV
#         lower_red1 = np.array([0, 120, 70])
#         upper_red1 = np.array([10, 255, 255])
#         lower_red2 = np.array([170, 120, 70])
#         upper_red2 = np.array([180, 255, 255])

#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             gray = cv2.medianBlur(gray, 5)  # Apply median filter

#             # Create a mask for red color
#             mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
#             mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
#             mask = cv2.bitwise_or(mask1, mask2)

#             # Apply the mask to the grayscale image
#             gray = cv2.bitwise_and(gray, gray, mask=mask)

#             U, V = HornSchunck(prev_gray, gray)
#             vis = drawOpticalflow(gray, U, V)

#             cv2.imshow("Horn-Schunck Optical Flow", vis)

#             prev_gray = gray
#             prev_hsv = hsv

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#     else:
#         print("Usage: python HornSchunck.py <video_file>")
#         sys.exit()

# if __name__ == "__main__":
#     warnings.filterwarnings("ignore")
#     main()

########################## work on the contour of the red cube #########################

# import cv2
# import numpy as np
# import sys
# import warnings
# from scipy.ndimage import convolve

# def HornSchunck(im1, im2, alpha=0.0001, N=10):
#     HSFilter = np.array([[1/12, 1/6, 1/12],
#                          [1/6,    0, 1/6],
#                          [1/12, 1/6, 1/12]], float)

#     im1 = im1.astype(np.float32)
#     im2 = im2.astype(np.float32)

#     U = np.zeros_like(im1)
#     V = np.zeros_like(im1)

#     [dx, dy, dt] = computeDerivatives(im1, im2)

#     for _ in range(N):
#         Un = convolve(U, HSFilter)
#         Vn = convolve(V, HSFilter)
#         derivatives = (dx * Un + dy * Vn + dt) / (alpha**2 + dx**2 + dy**2)
#         U = Un - dx * derivatives
#         V = Vn - dy * derivatives

#     return U, V

# def computeDerivatives(im1, im2):
#     kX = np.array([[-1, 1],
#                    [-1, 1]]) * 0.25
#     kY = np.array([[-1, -1],
#                    [1, 1]]) * 0.25
#     fx = convolve(im1, kX) + convolve(im2, kX)
#     fy = convolve(im1, kY) + convolve(im2, kY)
#     ft = convolve(im1, np.ones((2, 2)) * 0.25) + convolve(im2, -np.ones((2, 2)) * 0.25)
#     return fx, fy, ft

# def drawOpticalflow(img, U, V, step=15, threshold=500):
#     h, w = img.shape[:2]
#     y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
#     fx, fy = U[y, x], V[y, x]

#     lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
#     lines = np.int32(lines)

#     vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

#     for (x1, y1), (x2, y2) in lines:
#         cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
#         cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

#     contours = getMotionContours(U, V, threshold)
#     for contour in contours:
#         cv2.polylines(vis, [contour], isClosed=True, color=(0, 255, 255), thickness=2)

#     return vis

# def getMotionContours(U, V, threshold=2):
#     magnitude = np.sqrt(U**2 + V**2)
#     mask = np.zeros(U.shape, dtype=np.uint8)
#     mask[magnitude > threshold] = 255

#     kernel = np.ones((3, 3), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

# def main():
#     if True:
#         cap = cv2.VideoCapture(0)

#         success, prev_frame = cap.read()
#         if not success:
#             print("Error reading video file.")
#             cap.release()
#             sys.exit()

#         prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
#         prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#         prev_gray = cv2.medianBlur(prev_gray, 5)  # Apply median filter

#         # Define the range for red color in HSV
#         lower_red1 = np.array([0, 120, 70])
#         upper_red1 = np.array([10, 255, 255])
#         lower_red2 = np.array([170, 120, 70])
#         upper_red2 = np.array([180, 255, 255])

#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             gray = cv2.medianBlur(gray, 5)  # Apply median filter

#             # Create a mask for red color
#             mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
#             mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
#             mask = cv2.bitwise_or(mask1, mask2)

#             # Apply the mask to the grayscale image
#             gray = cv2.bitwise_and(gray, gray, mask=mask)

#             # Find contours of the red object
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#             # Draw contours on the original frame
#             for contour in contours:
#                 cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)  # Yellow color in BGR

#             U, V = HornSchunck(prev_gray, gray)
#             vis = drawOpticalflow(gray, U, V)

#             cv2.imshow("Horn-Schunck Optical Flow", vis)
#             cv2.imshow("Contours", frame)

#             prev_gray = gray
#             prev_hsv = hsv

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#     else:
#         print("Usage: python HornSchunck.py <video_file>")
#         sys.exit()

# if __name__ == "__main__":
#     warnings.filterwarnings("ignore")
#     main()


###########################"" work on the bleu face of the cube ~########################"""

# import cv2
# import numpy as np
# import sys
# import warnings
# from scipy.ndimage import convolve

# def HornSchunck(im1, im2, alpha=0.0001, N=10):
#     HSFilter = np.array([[1/12, 1/6, 1/12],
#                          [1/6,    0, 1/6],
#                          [1/12, 1/6, 1/12]], float)

#     im1 = im1.astype(np.float32)
#     im2 = im2.astype(np.float32)

#     U = np.zeros_like(im1)
#     V = np.zeros_like(im1)

#     [dx, dy, dt] = computeDerivatives(im1, im2)

#     for _ in range(N):
#         Un = convolve(U, HSFilter)
#         Vn = convolve(V, HSFilter)
#         derivatives = (dx * Un + dy * Vn + dt) / (alpha**2 + dx**2 + dy**2)
#         U = Un - dx * derivatives
#         V = Vn - dy * derivatives

#     return U, V

# def computeDerivatives(im1, im2):
#     kX = np.array([[-1, 1],
#                    [-1, 1]]) * 0.25
#     kY = np.array([[-1, -1],
#                    [1, 1]]) * 0.25
#     fx = convolve(im1, kX) + convolve(im2, kX)
#     fy = convolve(im1, kY) + convolve(im2, kY)
#     ft = convolve(im1, np.ones((2, 2)) * 0.25) + convolve(im2, -np.ones((2, 2)) * 0.25)
#     return fx, fy, ft

# def drawOpticalflow(img, U, V, step=15, threshold=500):
#     h, w = img.shape[:2]
#     y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
#     fx, fy = U[y, x], V[y, x]

#     lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
#     lines = np.int32(lines)

#     vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

#     for (x1, y1), (x2, y2) in lines:
#         cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
#         cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

#     contours = getMotionContours(U, V, threshold)
#     for contour in contours:
#         cv2.polylines(vis, [contour], isClosed=True, color=(0, 255, 255), thickness=2)

#     return vis

# def getMotionContours(U, V, threshold=2):
#     magnitude = np.sqrt(U**2 + V**2)
#     mask = np.zeros(U.shape, dtype=np.uint8)
#     mask[magnitude > threshold] = 255

#     kernel = np.ones((3, 3), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

# def main():
#     if True:
#         cap = cv2.VideoCapture(0)

#         success, prev_frame = cap.read()
#         if not success:
#             print("Error reading video file.")
#             cap.release()
#             sys.exit()

#         prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
#         prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#         prev_gray = cv2.medianBlur(prev_gray, 5)  # Apply median filter

#         # Define the range for blue color in HSV
#         lower_blue = np.array([100, 150, 0])
#         upper_blue = np.array([140, 255, 255])

#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             gray = cv2.medianBlur(gray, 5)  # Apply median filter

#             # Create a mask for blue color
#             mask = cv2.inRange(hsv, lower_blue, upper_blue)

#             # Apply the mask to the grayscale image
#             gray = cv2.bitwise_and(gray, gray, mask=mask)

#             # Find contours of the blue object
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#             # Draw contours on the original frame
#             for contour in contours:
#                 cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)  # Yellow color in BGR

#             U, V = HornSchunck(prev_gray, gray)
#             vis = drawOpticalflow(gray, U, V)

#             cv2.imshow("Horn-Schunck Optical Flow", vis)
#             cv2.imshow("Contours", frame)

#             prev_gray = gray
#             prev_hsv = hsv

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#     else:
#         print("Usage: python HornSchunck.py <video_file>")
#         sys.exit()

# if __name__ == "__main__":
#     warnings.filterwarnings("ignore")
#     main()

##################################### compute only on the point that are detected #################################################################

# import cv2
# import numpy as np
# import sys
# import warnings
# from scipy.ndimage import convolve

# def HornSchunck(im1, im2, contour_mask, alpha=0.001, N=10):
#     HSFilter = np.array([[1/12, 1/6, 1/12],
#                          [1/6,    0, 1/6],
#                          [1/12, 1/6, 1/12]], float)

#     im1 = im1.astype(np.float32)
#     im2 = im2.astype(np.float32)

#     U = np.zeros_like(im1)
#     V = np.zeros_like(im1)

#     [dx, dy, dt] = computeDerivatives(im1, im2)

#     for _ in range(N):
#         Un = convolve(U, HSFilter)
#         Vn = convolve(V, HSFilter)
#         derivatives = (dx * Un + dy * Vn + dt) / (alpha**2 + dx**2 + dy**2)
#         U = Un - dx * derivatives
#         V = Vn - dy * derivatives

#     # Apply the contour mask to the optical flow
#     U = U * contour_mask
#     V = V * contour_mask

#     return U, V

# def computeDerivatives(im1, im2):
#     kX = np.array([[-1, 1],
#                    [-1, 1]]) * 0.25
#     kY = np.array([[-1, -1],
#                    [1, 1]]) * 0.25
#     fx = convolve(im1, kX) + convolve(im2, kX)
#     fy = convolve(im1, kY) + convolve(im2, kY)
#     ft = convolve(im1, np.ones((2, 2)) * 0.25) + convolve(im2, -np.ones((2, 2)) * 0.25)
#     return fx, fy, ft

# def drawOpticalflow(img, U, V, step=30, threshold=1000):
#     h, w = img.shape[:2]
#     y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
#     fx, fy = U[y, x], V[y, x]

#     lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
#     lines = np.int32(lines)

#     vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

#     for (x1, y1), (x2, y2) in lines:
#         cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
#         cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

#     contours = getMotionContours(U, V, threshold)
#     for contour in contours:
#         cv2.polylines(vis, [contour], isClosed=True, color=(0, 255, 255), thickness=2)

#     return vis

# def getMotionContours(U, V, threshold=5):
#     magnitude = np.sqrt(U**2 + V**2)
#     mask = np.zeros(U.shape, dtype=np.uint8)
#     mask[magnitude > threshold] = 255

#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

# def main():
#     if True:
#         cap = cv2.VideoCapture(0)

#         success, prev_frame = cap.read()
#         if not success:
#             print("Error reading video file.")
#             cap.release()
#             sys.exit()

#         prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
#         prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#         prev_gray = cv2.medianBlur(prev_gray, 5)  # Apply median filter

#         # Define the range for blue color in HSV
#         lower_blue = np.array([100, 150, 0])
#         upper_blue = np.array([140, 255, 255])

#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             gray = cv2.medianBlur(gray, 5)  # Apply median filter

#             # Create a mask for blue color
#             mask = cv2.inRange(hsv, lower_blue, upper_blue)

#             # Apply the mask to the grayscale image
#             gray = cv2.bitwise_and(gray, gray, mask=mask)

#             # Find contours of the blue object
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#             # Create a mask for the contours
#             contour_mask = np.zeros_like(gray, dtype=np.uint8)
#             cv2.drawContours(contour_mask, contours, -1, (255), thickness=cv2.FILLED)

#             # Draw contours on the original frame
#             for contour in contours:
#                 cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)  # Yellow color in BGR

#             U, V = HornSchunck(prev_gray, gray, contour_mask)
#             vis = drawOpticalflow(gray, U, V)

#             cv2.imshow("Horn-Schunck Optical Flow", vis)
#             cv2.imshow("Contours", frame)

#             prev_gray = gray
#             prev_hsv = hsv

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#     else:
#         print("Usage: python HornSchunck.py <video_file>")
#         sys.exit()

# if __name__ == "__main__":
#     warnings.filterwarnings("ignore")
#     main()

############################# SIFT algorithm in the hsv plan #############################

# import cv2
# import numpy as np
# from scipy.ndimage import convolve

# def HornSchunck(im1, im2, contour_mask, alpha=0.001, N=10):
#     HSFilter = np.array([[1/12, 1/6, 1/12],
#                          [1/6,    0, 1/6],
#                          [1/12, 1/6, 1/12]], float)

#     im1 = im1.astype(np.float32)
#     im2 = im2.astype(np.float32)

#     U = np.zeros_like(im1)
#     V = np.zeros_like(im1)

#     [dx, dy, dt] = computeDerivatives(im1, im2)

#     for _ in range(N):
#         Un = convolve(U, HSFilter)
#         Vn = convolve(V, HSFilter)
#         derivatives = (dx * Un + dy * Vn + dt) / (alpha**2 + dx**2 + dy**2)
#         U = Un - dx * derivatives
#         V = Vn - dy * derivatives

#     # Apply the contour mask to the optical flow
#     U = U * contour_mask
#     V = V * contour_mask

#     return U, V

# def computeDerivatives(im1, im2):
#     kX = np.array([[-1, 1],
#                    [-1, 1]]) * 0.25
#     kY = np.array([[-1, -1],
#                    [1, 1]]) * 0.25
#     fx = convolve(im1, kX) + convolve(im2, kX)
#     fy = convolve(im1, kY) + convolve(im2, kY)
#     ft = convolve(im1, np.ones((2, 2)) * 0.25) + convolve(im2, -np.ones((2, 2)) * 0.25)
#     return fx, fy, ft

# def drawOpticalFlow(img, U, V, step=30):
#     h, w = img.shape[:2]
#     y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
#     fx, fy = U[y, x], V[y, x]

#     lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
#     lines = np.int32(lines)

#     vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     for (x1, y1), (x2, y2) in lines:
#         cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
#         cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
#     return vis

# def main():
#     cap = cv2.VideoCapture(1)

#     success, prev_frame = cap.read()
#     if not success:
#         print("Error: Cannot access the camera.")
#         return

#     prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
#     prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

#     sift = cv2.SIFT_create()

#     lower_color = np.array([100, 150, 0])  # Example HSV range for blue
#     upper_color = np.array([140, 255, 255])

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Create a mask for the target color
#         mask = cv2.inRange(hsv, lower_color, upper_color)
#         filtered_gray = cv2.bitwise_and(gray, gray, mask=mask)

#         # Detect keypoints using SIFT
#         keypoints, descriptors = sift.detectAndCompute(filtered_gray, None)
#         frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#         # Create a binary mask from contours
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contour_mask = np.zeros_like(gray, dtype=np.uint8)
#         cv2.drawContours(contour_mask, contours, -1, (255), thickness=cv2.FILLED)

#         # Compute optical flow
#         U, V = HornSchunck(prev_gray, gray, contour_mask)
#         optical_flow_visualization = drawOpticalFlow(gray, U, V)

#         # Display the results
#         cv2.imshow("Original with Keypoints", frame_with_keypoints)
#         cv2.imshow("Horn-Schunck Optical Flow", optical_flow_visualization)

#         prev_gray = gray.copy()

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

########################## modification of the arrow ##########################

# import cv2
# import numpy as np
# from scipy.ndimage import convolve

# def HornSchunck(im1, im2, contour_mask, alpha=0.001, N=10):
#     HSFilter = np.array([[1/12, 1/6, 1/12],
#                          [1/6,    0, 1/6],
#                          [1/12, 1/6, 1/12]], float)

#     im1 = im1.astype(np.float32)
#     im2 = im2.astype(np.float32)

#     U = np.zeros_like(im1)
#     V = np.zeros_like(im1)

#     [dx, dy, dt] = computeDerivatives(im1, im2)

#     for _ in range(N):
#         Un = convolve(U, HSFilter)
#         Vn = convolve(V, HSFilter)
#         derivatives = (dx * Un + dy * Vn + dt) / (alpha**2 + dx**2 + dy**2)
#         U = Un - dx * derivatives
#         V = Vn - dy * derivatives

#     # Apply the contour mask to the optical flow
#     U = U * contour_mask
#     V = V * contour_mask

#     return U, V

# def computeDerivatives(im1, im2):
#     kX = np.array([[-1, 1],
#                    [-1, 1]]) * 0.25
#     kY = np.array([[-1, -1],
#                    [1, 1]]) * 0.25
#     fx = convolve(im1, kX) + convolve(im2, kX)
#     fy = convolve(im1, kY) + convolve(im2, kY)
#     ft = convolve(im1, np.ones((2, 2)) * 0.25) + convolve(im2, -np.ones((2, 2)) * 0.25)
#     return fx, fy, ft

# def drawOpticalFlow(img, U, V, step=30, scale=10):
#     h, w = img.shape[:2]
#     y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
#     fx, fy = U[y, x], V[y, x]

#     # Normalize the optical flow vectors
#     magnitude = np.sqrt(fx**2 + fy**2)
#     fx = fx / magnitude * scale
#     fy = fy / magnitude * scale

#     lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
#     lines = np.int32(lines)

#     vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     for (x1, y1), (x2, y2) in lines:
#         cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
#         cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
#     return vis

# def main():
#     # Change the index to switch to the external webcam
#     cap = cv2.VideoCapture(1)  # Use 1 for the first external webcam, 2 for the second, etc.

#     success, prev_frame = cap.read()
#     if not success:
#         print("Error: Cannot access the camera.")
#         return

#     prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
#     prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

#     sift = cv2.SIFT_create()

#     lower_color = np.array([100, 150, 0])  # Example HSV range for blue
#     upper_color = np.array([140, 255, 255])

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Create a mask for the target color
#         mask = cv2.inRange(hsv, lower_color, upper_color)
#         filtered_gray = cv2.bitwise_and(gray, gray, mask=mask)

#         # Detect keypoints using SIFT
#         keypoints, descriptors = sift.detectAndCompute(filtered_gray, None)
#         frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#         # Create a binary mask from contours
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contour_mask = np.zeros_like(gray, dtype=np.uint8)
#         cv2.drawContours(contour_mask, contours, -1, (255), thickness=cv2.FILLED)

#         # Compute optical flow
#         U, V = HornSchunck(prev_gray, gray, contour_mask)
#         optical_flow_visualization = drawOpticalFlow(gray, U, V)

#         # Display the results
#         cv2.imshow("Original with Keypoints", frame_with_keypoints)
#         cv2.imshow("Horn-Schunck Optical Flow", optical_flow_visualization)
#         cv2.imshow("f", filtered_gray)

#         prev_gray = gray.copy()

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

#todo appliquer horn schunck uniquement sur les points detecter par sift
#todo ajouter un filtrage a sift pour réduire le nombre de détection
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
    """Main function to track blue cube using SIFT and sparse optical flow"""
    cap = cv2.VideoCapture(0)
    sift = cv2.SIFT_create()
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))

    prev_frame = None
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
            roi = frame[y:y+h, x:x+w]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Detect keypoints
            kp = sift.detect(gray_roi, None)

            if prev_frame is not None and len(kp) > 0:
                prev_pts = np.float32([kp.pt for kp in kp])

                # Ensure the ROI dimensions are the same
                if prev_frame.shape != gray_roi.shape:
                    prev_frame = cv2.resize(prev_frame, (gray_roi.shape[1], gray_roi.shape[0]))

                # Create pyramid levels explicitly
                prev_pyr = buildOpticalFlowPyramid(prev_frame, maxLevel=2)
                curr_pyr = buildOpticalFlowPyramid(gray_roi, maxLevel=2)

                # Compute optical flow
                next_pts, pts_status, pts_error = cv2.calcOpticalFlowPyrLK(
                    prev_pyr[0], curr_pyr[0],
                    prev_pts, None, **lk_params)

                # Draw tracks
                for i, (new, old) in enumerate(zip(next_pts, prev_pts)):
                    if pts_status[i]:
                        a, b = new.ravel()
                        cv2.circle(roi, (int(a), int(b)), 2, (0, 255, 0), -1)

            # Store current frame for next iteration
            prev_frame = gray_roi.copy()
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Cube Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def buildOpticalFlowPyramid(image, maxLevel=2):
    """Build pyramid levels for optical flow"""
    temp = [image]
    for i in range(maxLevel):
        temp.append(cv2.pyrDown(temp[-1]))
    return temp

if __name__ == "__main__":
    track_cube()

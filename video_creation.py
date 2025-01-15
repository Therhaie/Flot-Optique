import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the codec and create a VideoWriter object
# Use 'MPEG' as the codec for .mpeg format (or 'XVID', 'MP4V', depending on the system)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # FourCC for .mpeg-compatible codec
out = cv2.VideoWriter('output.mpeg', fourcc, 20.0, (640, 480))

print("Recording video. Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Write the frame to the output video file
    out.write(frame)

    # Show the frame
    cv2.imshow('Recording', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

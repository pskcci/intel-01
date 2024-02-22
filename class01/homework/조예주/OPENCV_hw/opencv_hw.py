import numpy as np
import cv2

# Open the default camera
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Couldn't open the camera.")
    exit()

# Loop to read frames from the camera
while cap.isOpened():
    # Read a frame from the camera
    frame = cap.read()[1]  # Only get the frame, ignore the success flag
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get a binary image
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
   
    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=40, minRadius=5, maxRadius=70)
    
    # Check if circles are detected
    if circles is not None:
        # Convert the circle parameters (x, y, radius) to integers
        circles = np.round(circles[0, :]).astype("int")
        
        # Loop over all detected circles and draw them on the frame
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)  # Draw the circle
        
    # Display the frame with detected circles
    cv2.imshow("Circles Detected", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
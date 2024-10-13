import cv2

# Load cascades
nose_cascade = cv2.CascadeClassifier('./third-party/haarcascade_mcs_nose.xml')
eye_cascade = cv2.CascadeClassifier('./third-party/frontalEyes35x16.xml')

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect nose and eyes
    nose = nose_cascade.detectMultiScale(gray_frame, 1.3, 5)
    eyes = eye_cascade.detectMultiScale(gray_frame, 1.3, 5)
    
    # Draw rectangles around detected features
    for (x, y, w, h) in nose:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the output frame
    cv2.imshow("Video Frame", frame)
    
    # Exit on pressing 'q'
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

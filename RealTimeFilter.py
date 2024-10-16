import cv2
import numpy as np

nose_cascade = cv2.CascadeClassifier('./third-party/haarcascade_mcs_nose.xml')
eye_cascade = cv2.CascadeClassifier('./third-party/frontalEyes35x16.xml')

sunglasses = cv2.imread('glasses.png', -1)
moustache = cv2.imread('mustache.png', -1)

cap = cv2.VideoCapture(0)

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`."""
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])


    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    alpha = alpha_mask[y1o:y2o, x1o:x2o] / 255.0
    alpha_inv = 1.0 - alpha

    for c in range(0, 3):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    nose = nose_cascade.detectMultiScale(gray_frame, 1.3, 5)
    eyes = eye_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (nx, ny, nw, nh) in nose:
        moustache_resized = cv2.resize(moustache, (nw, int(nh / 2)))
        overlay_image_alpha(frame, moustache_resized, nx, ny + int(nh / 1.5), moustache_resized[:, :, 3])
        break

    for (ex, ey, ew, eh) in eyes:
        sunglasses_resized = cv2.resize(sunglasses, (ew * 2, eh))
        overlay_image_alpha(frame, sunglasses_resized, ex - ew // 2, ey, sunglasses_resized[:, :, 3])
        break
    
    cv2.imshow("Video Frame", frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

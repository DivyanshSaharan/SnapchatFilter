import cv2
import pandas as pd
import numpy as np

input_image_path = 'Before.png'
image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)

if image is None:
    raise ValueError("Could not load the input image. Please check the file path.")

print(f"Original image shape: {image.shape}")

sunglasses = cv2.imread('glasses.png', -1)  
moustache = cv2.imread('mustache.png', -1)  

if sunglasses is None:
    raise ValueError("Could not load sunglasses image. Please check the file path.")
if moustache is None:
    raise ValueError("Could not load mustache image. Please check the file path.")

nose_cascade = cv2.CascadeClassifier('./third-party/haarcascade_mcs_nose.xml')
eye_cascade = cv2.CascadeClassifier('./third-party/frontalEyes35x16.xml')

if nose_cascade.empty() or eye_cascade.empty():
    raise ValueError("Could not load Haar cascade classifiers. Please check the file paths.")

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

gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

nose = nose_cascade.detectMultiScale(gray_frame, 1.3, 5)
eyes = eye_cascade.detectMultiScale(gray_frame, 1.3, 5)

print(f'Detected noses: {nose}')
print(f'Detected eyes: {eyes}')

for (nx, ny, nw, nh) in nose:
    moustache_resized = cv2.resize(moustache, (nw, int(nh / 2)))
    overlay_image_alpha(image, moustache_resized, nx, ny + int(nh / 2), moustache_resized[:, :, 3])
    break

for (ex, ey, ew, eh) in eyes:
    sunglasses_resized = cv2.resize(sunglasses, (ew * 2, eh))
    overlay_image_alpha(image, sunglasses_resized, ex - ew // 2, ey - eh // 2, sunglasses_resized[:, :, 3])
    break

cv2.imshow("Modified Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

if image.shape[2] == 4:
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

modified_pixel_values = image.reshape(-1, 3)

print(f"Modified image shape for CSV: {modified_pixel_values.shape}")

modified_df = pd.DataFrame(modified_pixel_values, columns=['Blue', 'Green', 'Red'])

result_csv_path = 'result.csv'
modified_df.to_csv(result_csv_path, index=False)

print(f'Result saved to {result_csv_path}')

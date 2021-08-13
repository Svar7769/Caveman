# Importing the libraries
import cv2
import numpy as np

from pre import prediction

minValue = 70

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()

    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])

    # Drawing the ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
    # Extracting the ROI
    roi = frame[y1:x1, y2:x2].copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Applying blur
    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    # reducing feature from the images
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    test_image = cv2.resize(test_image, (300, 300))
    cv2.imshow("test", test_image)

    if cv2.waitKey(1) & 0xFF == ord('1'):
        cv2.imwrite('image.jpg', roi)
        pri = prediction('image.jpg')
        cv2.putText(frame, str(pri), (10, 410), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Importing the libraries
import keras
from keras.models import load_model
import cv2
import numpy as np

Dict = {0: 'A',1: 'B',2: 'C',3: 'D',4: 'E',5: 'F',6: 'G',7: 'H',8: 'I',9: 'J',10: 'K',11: 'L',12: 'M',13: 'N',14: 'O',
    15: 'P',16: 'Q',17: 'R',18: 'S',19: 'T',20: 'U',21: 'V',22: 'W',23: 'X',24: 'Y',25: 'Z',26: 'Del',27: 'Nothing',3: 'Space'
    }

minValue = 70

model = load_model('save_model')

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()

    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])

    # Drawing the ROI
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255, 0, 0), 1)
    # Extracting the ROI
    roi = frame[y1:x1, y2:x2].copy()
    test_image = cv2.resize(roi, (64, 64))

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _,test_image = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)

    # # Applying blur
    # blur = cv2.GaussianBlur(gray, (5, 5), 2)
    #
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5, 5), 2)
    #
    # th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow("test", test_image)
    test_images = np.reshape(test_image,[1,64,64,3])
    # Prediction
    result = model.predict_classes(test_images)

    prediction = Dict[int(result)]

    cv2.putText(frame,prediction, (x1+100, y2+30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1)

    # if cv2.waitKey(1) & 0xFF == ord('1'):
    #     dim = (64,64)
    #     img = cv2.resize(img, (64, 64))
    #
    #     img = np.reshape(img, [1, 64, 64, 3])
    #
    #     classes = model.predict_classes(img)
    #
    #     print(Dict[int(classes)])
    #
    #     cv2.putText(frame, str(classes), (10, 410), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

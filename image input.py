import keras
from keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf

# Using Dictionary to get output from testing import prediction

Dict = {0: 'A',1: 'B',2: 'C',3: 'D',4: 'E',5: 'F',6: 'G',7: 'H',8: 'I',9: 'J',10: 'K',11: 'L',12: 'M',13: 'N',14: 'O',
    15: 'P',16: 'Q',17: 'R',18: 'S',19: 'T',20: 'U',21: 'V',22: 'W',23: 'X',24: 'Y',25: 'Z',26: 'Del',27: 'Nothing',28: 'Space'
    }

# loading Deep learning model
model = load_model('save_model_Color')

# Capture Video
cap = cv2.VideoCapture(0)

# to get Constant feedback
while True:
    # reading capture Video
    _, frame = cap.read()

    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    # cordinate of ROI

    x1 = int(0.7 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.3 * frame.shape[1])

    # Drawing the ROI
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)

    # Extracting the ROI
    roi = frame[y1:y2, x1:x2].copy()

    cv2.imshow("test", roi)

    # resizing ROI to 64X64
    roi = cv2.resize(roi,(64,64))

    #reshaping
    img = np.reshape(roi,[1,64,64,3])

    # pedicting Class
    classes = model.predict_classes(img)

    # printing class
    print(classes)
    # Relating class with value using Dict
    print(Dict[int(classes)])
    #Storing Result
    prediction = Dict[int(classes)]
    cv2.putText(frame, prediction, (x1 + 100, y2 + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1)

    #Show frame
    cv2.imshow("Frame", frame)
    # to close window press ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
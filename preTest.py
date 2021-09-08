import cv2
from PIL import Image, ImageOps
from keras.models import load_model
from keras.saving.model_config import model_from_json
import numpy as np
from keras_preprocessing.image import img_to_array


Dict = {0: 'A',1: 'B',2: 'C',3: 'D',4: 'E',5: 'F',6: 'G',7: 'H',8: 'I',9: 'J',10: 'K',11: 'L',12: 'M',13: 'N',14: 'O',
    15: 'P',16: 'Q',17: 'R',18: 'S',19: 'T',20: 'U',21: 'V',22: 'W',23: 'X',24: 'Y',25: 'Z',26: 'Del',27: 'Nothing',3: 'Space'
    }

image = 'data/train/del/del100.jpg'

model = load_model('save_model')

img = cv2.imread(image)

img = cv2.resize(img,(64,64))

img = np.reshape(img,[1,64,64,3])

classes = model.predict_classes(img)

print(classes)

print (Dict[int(classes)])

import cv2
from PIL import Image, ImageOps
from keras.models import load_model
from keras.saving.model_config import model_from_json
import numpy as np
from keras_preprocessing.image import img_to_array

from image_processing import func

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

def prediction(image):

    test_image = func(image)
    cv2.imwrite("image.jpg", test_image)

    image = Image.open('image.jpg')
    image = image.resize((128, 128))
    X = img_to_array(image)
    X = np.expand_dims(X,axis = 0)

    pred = model.predict(X)
    max_index_row = np.argmax(pred, axis=1)
    print(max_index_row)

    return max_index_row
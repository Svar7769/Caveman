# Importing the Keras libraries and packages
import random

import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
import numpy as np
import tensorflow as tp
import os

#use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
base_dir = "data/train"
sz = 128

# Building the CNN

# Initializing the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Second convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.10))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.10))
classifier.add(Dense(units=64, activation='relu'))
# Softmax to classify more than 2
classifier.add(Dense(units=35, activation='softmax'))

# Compiling the CNN
# categorical_crossentropy for more than 2
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Preparing the train/test data and training the model
classifier.summary()

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.1
)
test_datagen = ImageDataGenerator(rescale=1./255,
                                  validation_split=0.1
)

train_datagen = train_datagen.flow_from_directory(
        base_dir,
        target_size=(sz, sz),
        batch_size=10,
        subset='training',
        color_mode='grayscale',
        class_mode='categorical'
)
test_datagen = test_datagen.flow_from_directory(
        base_dir,
        target_size=(sz, sz),
        batch_size=10,
        subset='validation',
        color_mode='grayscale',
        class_mode='categorical')

history = classifier.fit(
        train_datagen,
        steps_per_epoch=int(3077/10),
        epochs=40,
        validation_data=test_datagen,
        validation_steps=int(329/10)
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Saving the model
#model detain in jason file
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
classifier.save_weights('model.h5')
print('Weights saved')

pre = classifier.predict('images/8/0.jpg')
print(pre)

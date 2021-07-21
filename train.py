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
from matplotlib import image as mpimg

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
classifier.add(Dropout(0.40))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=64, activation='relu'))
# Softmax to classify more than 2
classifier.add(Dense(units=27, activation='softmax'))

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
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/test',
        target_size=(sz, sz),
        batch_size=10,
        color_mode='grayscale',
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(sz, sz),
        batch_size=10,
        color_mode='grayscale',
        class_mode='categorical')

classifier.fit(
        train_generator,
        steps_per_epoch=30,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=800)

# classifier.fit(
#     training_set,
#     steps_per_epoch=30,  # No of images in training set
#     epochs=5,
#     validation_data=test_set,
#     validation_steps=2524)  # No of images in test set

# Saving the model
#model detain in jason file
model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
classifier.save_weights('model-bw.h5')
print('Weights saved')

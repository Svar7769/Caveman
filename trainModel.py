# Importing the Keras libraries and packages
import random

import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
import os

diretorio = 'data/train'

SIZE=64
image_size = (SIZE, SIZE)
batch_size = 32
channels = 1
kernel = 4
stride = 1
pool = 2

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    diretorio,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    diretorio,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
)

class_names = train_ds.class_names
num_classes = len(class_names)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# model from scratch
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(64, 64, 1)),
    layers.Conv2D(16, kernel, stride, activation='relu'),
    layers.MaxPool2D(pool),
    layers.Conv2D(32, kernel, stride, activation='relu'),
    layers.MaxPool2D(pool),
    layers.Conv2D(64, kernel, stride, activation='relu'),
    layers.MaxPool2D(pool),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoint.h5',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

#treinando
epochs = 17
history = model.fit(
    train_ds, epochs=epochs, callbacks=model_checkpoint_callback, validation_data=val_ds,
)

model.summary()

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

model.save('save_model_D1')

print("Model Saved")
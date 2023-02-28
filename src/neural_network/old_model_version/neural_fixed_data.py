import os
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt

from keras.layers import preprocessing

from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16

from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg16 import preprocess_input

train_dir = os.path.join('data/fixed_data/train')
test_dir = os.path.join('data/fixed_data/test')
val_dir = os.path.join('data/fixed_data/val')

image_size = (448, 448)
fine_tune_at_procent = 0.2
batch_size = 32
autotune = tf.data.AUTOTUNE
input_shape = image_size + (3, )

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, shuffle=True, batch_size=batch_size, image_size=image_size)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir, batch_size=batch_size, image_size=image_size)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir, shuffle=True, batch_size=batch_size, image_size=image_size)

print(len(test_dataset.class_names))

model = VGG16(input_shape=input_shape, include_top=False, weights='imagenet', number_neurons=7)
model.trainable = True
fine_tune_at = int(len(model.layers) * fine_tune_at_procent)

for layer in model.layers[:fine_tune_at]:
    layer.trainable = False
data_aug_layers = tf.keras.Sequential([tf.keras.layers.RandomRotation(0.2),
                                       tf.keras.layers.RandomBrightness(0.2),
                                       tf.keras.layers.RandomFlip('horizontal')])

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
gap = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(
    len(test_dataset.class_names), activation='softmax')
inputs = tf.keras.Input(shape=input_shape)

x = data_aug_layers(inputs)
x = preprocess_input(x)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.4)(x)

outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

history = model.fit(train_dataset, epochs=100, validation_data=val_dataset)

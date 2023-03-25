import os
import tensorflow as tf
import pandas as pd
import keras
import datetime

from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.resnet_v2 import ResNet50V2

from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet_v2 import preprocess_input

log_dir = 'data/logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

train_dir = os.path.join('data/nonfixed_data/train')
val_dir = os.path.join('data/nonfixed_data/val')
test_dir = os.path.join('data/nonfixed_data/test')

AUTOTUNE = tf.data.AUTOTUNE

BATCH_SIZES = 72
EPOCHS = 100
IMG_SIZES = (224, 224)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, batch_size = BATCH_SIZES, image_size = IMG_SIZES)

val_dataset = tf.keras.utils.image_dataset_from_directory(val_dir, shuffle=True, batch_size = BATCH_SIZES, image_size = IMG_SIZES)

test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir, shuffle=False, batch_size = BATCH_SIZES, image_size = IMG_SIZES)

IMG_SHAPE = IMG_SIZES + (3, )

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

from prompt_toolkit import output
from tensorflow.python.ops.gen_array_ops import shape


preprocessing_for_models = keras.applications.resnet_v2.preprocess_input
def create_model(VGG16, preprocessing, data_aug_layers=None, INPUT_SHAPE: tuple = (224, 224, 3), fine_tune_at_procent:float = 0.5, num_classes: int = 7):

  VGG16 = VGG16(input_shape = INPUT_SHAPE, include_top=False, weights='imagenet')
  VGG16.trainable = True



  fine_tune_at = int(len(VGG16.layers) * fine_tune_at_procent)

  for layer in VGG16.layers[:fine_tune_at]:
    layer.trainable = False

  resacle = tf.keras.layers.Rescaling(1./127.5, offset=-1)

  if data_aug_layers is None:

    data_aug_layers = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomFlip('horizontal')

    ])

  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

  prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

  inputs = tf.keras.Input(shape=INPUT_SHAPE)
  # x = data_aug_layers(inputs)
  x = preprocessing(inputs)
  x = VGG16(x, training=False)
  x = global_average_layer(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  outputs = prediction_layer(x)
  model = tf.keras.Model(inputs, outputs)
  
  return model



current_model = create_model(ResNet50V2, preprocessing_for_models, INPUT_SHAPE=IMG_SHAPE, fine_tune_at_procent=0.2)
current_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
current_model.summary()
history = current_model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=[tensorflow_callback])
current_model.save('data/model/my_model.h5py')
pd.DataFrame.from_dict(history.history).to_csv(os.path.join('data/model/history.csv'), index=False)


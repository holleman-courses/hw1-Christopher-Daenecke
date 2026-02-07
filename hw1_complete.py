#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


## 

def build_model1():
  model = tf.keras.Sequential([
    layers.Flatten(input_shape=(32,32,3)),
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(10)
  ]) 
  
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  model.summary()

  return model

def build_model2():
  model = tf.keras.Sequential([
    layers.Conv2D(32, kernel_size=(3,3), strides=(2,2), padding="same", activation="relu", input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.Conv2D(64, kernel_size=(3,3), strides=(2,2), padding="same", activation="relu"),
    layers.BatchNormalization(),

    layers.Conv2D(128, kernel_size=(3,3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3,3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3,3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3,3), padding="same", activation="relu"),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(10)
  ])

  
  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  model.summary()
  
  return model

def build_model3():
  model = None # Add code to define model 1.
  ## This one should use the functional API so you can create the residual connections
  model = tf.keras.Sequential([
    layers.SeparableConv2D(32, kernel_size=(3,3), strides=(2,2), padding="same", activation="relu", input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.SeparableConv2D(64, kernel_size=(3,3), strides=(2,2), padding="same", activation="relu"),
    layers.BatchNormalization(),

    layers.SeparableConv2D(128, kernel_size=(3,3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size=(3,3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size=(3,3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size=(3,3), padding="same", activation="relu"),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(10)
  ])
  
  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  model.summary()

  return model

def build_model50k():
  model = tf.keras.Sequential([
    Input(shape=(32,32,3)),

    layers.Conv2D(32, kernel_size=(3,3), strides=(2,2), padding="same"),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Dropout(0.1),

    layers.SeparableConv2D(64, kernel_size=(3,3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.SeparableConv2D(128, kernel_size=(3,3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.SeparableConv2D(128, kernel_size=(3,3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(10)
  ]) 

  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  model.summary()
  return model

model50k = build_model50k()

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
  
  val_frac = 0.1
  num_val = int(len(train_images) * val_frac)

  val_idxs = np.random.choice(np.arange(len(train_images)), size=num_val, replace=False)
  trn_idxs = np.setdiff1d(np.arange(len(train_images)), val_idxs)

  train_images = train_images.astype("float32") / 255.0
  test_images = test_images.astype("float32") / 255.0

  val_images = train_images[val_idxs]
  val_labels = train_labels[val_idxs]
  train_images = train_images[trn_idxs]
  train_labels = train_labels[trn_idxs]

  ########################################
  ## Build and train model 1
  
  model1 = build_model1()

  history1 = model1.fit(
    train_images,
    train_labels,
    epochs = 30,
    validation_data = (val_images, val_labels),
  )
  
  train_acc1 = history1.history['accuracy'][-1]
  val_acc1 = history1.history['val_accuracy'][-1]

  test_loss1, test_acc1 = model1.evaluate(test_images, test_labels, verbose = 0)

  print(f"Model 1 final training accuracy: {train_acc1:.4f}")
  print(f"Model 1 final validation accuracy: {val_acc1:.4f}")
  print(f"Model 1 test accuracy: {test_acc1:.4f}")

  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()

  history2 = model2.fit(
    train_images,
    train_labels,
    epochs = 30,
    validation_data = (val_images, val_labels),
  )
  
  train_acc2 = history2.history['accuracy'][-1]
  val_acc2 = history2.history['val_accuracy'][-1]

  test_loss2, test_acc2 = model2.evaluate(test_images, test_labels, verbose = 0)

  print(f"Model 2 final training accuracy: {train_acc2:.4f}")
  print(f"Model 2 final validation accuracy: {val_acc2:.4f}")
  print(f"Model 2 test accuracy: {test_acc2:.4f}")

  test_img = np.array(keras.utils.load_img(
      './test_image_dog.jpg',
      grayscale=False,
      color_mode='rgb',
      target_size=(32,32)))

  test_img = test_img.astype("float32") / 255.0
  test_img = np.expand_dims(test_img, axis=0)

  output_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

  prediction = model2.predict(test_img)
  pred_class = np.argmax(prediction, axis=1)[0]

  print("Predicted class: ", output_classes[pred_class])

  ### Repeat for model 3 and your best sub-50k params model
  
  model3 = build_model3()

  history3 = model3.fit(
    train_images,
    train_labels,
    epochs = 30,
    validation_data = (val_images, val_labels),
  )

  train_acc3 = history3.history['accuracy'][-1]
  val_acc3 = history3.history['val_accuracy'][-1]

  test_loss3, test_acc3 = model3.evaluate(test_images, test_labels, verbose = 0)

  print(f"Model 3 final training accuracy: {train_acc3:.4f}")
  print(f"Model 3 final validation accuracy: {val_acc3:.4f}")
  print(f"Model 3 test accuracy: {test_acc3:.4f}")

  

  history50k = model50k.fit(
    train_images,
    train_labels,
    epochs = 30,
    validation_data = (val_images, val_labels),
  )

  model50k.save("best_model.h5")
  
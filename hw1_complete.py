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
      layers.Dense(128, activation=tf.keras.layers.LeakyReLU()),
      layers.Dense(128, activation=tf.keras.layers.LeakyReLU()),
      layers.Dense(128, activation=tf.keras.layers.LeakyReLU()),
      layers.Dense(10)
  ]) 
  
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  model.summary()

  return model

def build_model2():
  model = None # Add code to define model 1.
  return model

def build_model3():
  model = None # Add code to define model 1.
  ## This one should use the functional API so you can create the residual connections
  return model

def build_model50k():
  model = None # Add code to define model 1.
  return model

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
    batch_size = 32
  )
  # compile and train model 1.

  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()

  
  ### Repeat for model 3 and your best sub-50k params model
  
  

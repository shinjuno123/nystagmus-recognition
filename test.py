import tensorflow as tf
import json
import numpy as np
import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
 
path = "C:\\Users\\shinj\\Desktop\\nystagmus\\"
def getModel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96,return_sequences=True), input_shape=(15, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1024,activation = "relu"))
    model.add(tf.keras.layers.Dense(512,activation = "relu"))
    model.add(tf.keras.layers.Dense(128,activation = "relu"))
    model.add(tf.keras.layers.Dense(5,activation = "softmax"))
    opt = tf.keras.optimizers.Adam(learning_rate=0.0004)
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=opt, metrics=["accuracy"])
    model.save(path+"Classifier.h5")
    model.summary()
    return model

def load_data():
  x_train = np.load(path+'x_train.npy',allow_pickle=True)
  y_train = np.load(path+'y_train.npy',allow_pickle=True)
  x_test = np.load(path+'x_test.npy',allow_pickle=True)
  y_test = np.load(path+'y_test.npy',allow_pickle=True)
  print(x_train.shape, y_train.shape)
  print(x_test.shape, y_test.shape)
  return x_train,x_test,y_train,y_test 

model = getModel()
x_train, x_test, y_train, y_test = load_data()

earlystop = EarlyStopping(patience=7)
callbacks = [earlystop]
early_stopping = keras.callbacks.EarlyStopping(monitor="loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="loss", patience=5)

# Define modifiable training hyperparameters.
epochs = 300
batch_size = 32

# Fit the model to the training data.
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr],
    validation_data=(x_test, y_test)
)
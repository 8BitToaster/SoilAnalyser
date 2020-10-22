import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import random
import warnings
import random
import numpy as np
from keras.layers import Activation, Dense

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


#Different inputs:
#Ball test: 0 would mean high sand and 1 would be the opposite

#Ribbon test:
#0: No wire, is Sandy Loam
#x < 0.5: Less then 2.5 cm Loam column
#0.5 < x < 1: 2.5-5 cm Clay column
#1: 5cm+ Clay column

#Feel test (2 Variables):
#0-1: Soil feel gritty
#0-1: Soil feel smooth


#Classifiers:
#0.1: Sandy Loam
#0.2: Silt Loam
#0.3: Loam
#0.4: Sandy Clay Loam
#0.5: Silty Clay Loam
#0.6: Clay Loam
#0.7: Sandy Clay
#0.8: Silty Clay
#0.9: Clay

def genModel():
        inputs = keras.Input(shape=(None, 1,5))

        dense = layers.Dense(100, activation="relu")
        x = dense(inputs)
        x = layers.Dense(50, activation="relu")(x)
        x = dense(inputs)
        x = layers.Dense(25, activation="relu")(x)

        outputs = layers.Dense(1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

        model.compile(optimizer='sgd', loss='mean_squared_error')

        return model

model = genModel()
model.load_weights('./checkpoints/my_checkpoint')

Classifiers = {0.1: "Sandy Loam", 0.2: "Silt Loam", 0.3: "Loam", 0.4: "Sandy Clay Loam", 0.5: "Silty Clay Loam", 0.6: "Clay Loam", 0.7: "Sandy Clay", 0.8: "Silty Clay", 0.9: "Clay"}

while True:
        ballTest = min(max(float(input("From 0 to 100, how well did the ball hold: "))/100, 0), 1)
        ribbonTest = max(min(float(input("How long was the dirt ribbon in inches: "))*2.5/5, 1), 0)

        gritTest = min(max(float(input("From 0 to 100, how gritty was the soil: "))/100, 0), 1)
        smoothTest = min(max(float(input("From 0 to 100, how gritty was the soil: "))/100, 0), 1)

        values = []
        for value in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                values.append(model.predict([[ballTest, ribbonTest, gritTest, smoothTest, value]]))

        value, valueIndex = max(values), values.index(max(values))

        value = value[0][0]

        print("The model is: " + str(min(max(round(value*100, 2), 0), 100)) + "% sure the soil is " + Classifiers[(valueIndex+1)/10])
        


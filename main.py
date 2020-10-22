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


#Different inputs:
#Ball test: 0 would mean high sand and 1 would be the opposite

#Ribbon test:
#0: No wire, is Sandy Loam
#x > 0.5: Less then 2.5 cm Loam column
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

def genDataset(num):
    trainingX, trainingY = [], []
    Classifiers = {0.1: "Sandy Loam", 0.2: "Silt Loam", 0.3: "Loam", 0.4: "Sandy Clay Loam", 0.5: "Silty Clay Loam", 0.6: "Clay Loam", 0.7: "Sandy Clay", 0.8: "Silty Clay", 0.9: "Clay"}
    datasetTests = {0.1: [0, 0, 1, 0], 0.2: [1, 0.15, 0, 1], 0.3: [1, 0.15, 0, 0], 0.4: [0.3, 0.75, 1, 0], 0.5: [1, 0.75, 0, 1], 0.6: [1, 0.75, 0, 0],
                    0.7: [0.3, 1, 1, 0], 0.8: [1, 1, 0, 1], 0.9: [1, 1, 0, 0]}

    for i in range(num):
        num = random.randint(1, 9)/10

        testSet = datasetTests[num]

        for value in testSet:
            value = max(min(value + random.randint(-value*100, value*100)/1000, 1), 0)

        for value in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            testSet.append(value)
            trainingX.append(np.array(testSet))
            if num == value:
                trainingY.append(1)
            else:
                trainingY.append(0)

            testSet.pop(-1)


        

    return [trainingX, trainingY]

def genModel():
        inputs = keras.Input(shape=(1,5))
 
        dense = layers.Dense(100, activation="relu")
        x = dense(inputs)
        x = layers.Dense(50, activation="relu")(x)
        x = dense(inputs)
        x = layers.Dense(25, activation="relu")(x)
        
        outputs = layers.Dense(1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

        model.compile(optimizer='sgd', loss='mean_squared_error')

        return model


dataset = genDataset(5000)

x = np.array(dataset[0], dtype=float)
y = np.array(dataset[1], dtype=float)

model = genModel()
model.fit(x, y, epochs=1000)


model.save_weights('./checkpoints/my_checkpoint')

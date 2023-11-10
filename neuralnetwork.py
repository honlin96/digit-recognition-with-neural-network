# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:41:41 2020

@author: honlin
"""

import tensorflow as tf

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,x_test = x_train /255.0, x_test/255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation=tf.nn.relu) , #first layer, 512 nodes and relu activation function
    tf.keras.layers.Dropout(0.2), #randomly drop out 20% of the nodes
    tf.keras.layers.Dense(128,activation=tf.nn.relu), #2nd layer, 128 nodes and relu activation function
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
    ])
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train,y_train,epochs = 10) #epoch = full cycles
#evaluate the neural network
model.evaluate(x_test,y_test)
#predict the label of the image based on the trained neural network
#predictions = model.predict(test_images)
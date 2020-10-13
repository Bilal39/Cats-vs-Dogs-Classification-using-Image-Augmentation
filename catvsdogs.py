#Importing important libraries

import tensorflow as tf
import numpy as np
from tensorflow.keras import models,layers,Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Using Image data generator to rescale and image augmentation

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,rotation_range=40,
                                 horizontal_flip=True,width_shift_range=0.2,height_shift_range=0.2,fill_mode='nearest')
test_datagen=ImageDataGenerator(rescale=1./255)

train_dir= "C:/Users/MBK/Desktop/train2/train"
valid_dir= "C:/Users/MBK/Desktop/train2/test"

train_gen=train_datagen.flow_from_directory(train_dir,target_size=(150,150),class_mode='binary')
test_gen=test_datagen.flow_from_directory(valid_dir,target_size=(150,150),class_mode='binary')

#Model Architecture

model=models.Sequential([
                         Conv2D(64,(3,3),input_shape=(150,150,3),activation='relu'),
                         Dropout(0.3),
                         MaxPooling2D((2,2)),
                         BatchNormalization(),
                         Conv2D(32,(3,3),activation='relu'),
                         MaxPooling2D((2,2)),
                         BatchNormalization(),
                         Conv2D(32,(3,3),activation='relu'),
                         MaxPooling2D((2,2)),
                         BatchNormalization(),
                         Flatten(),
                         Dense(64,activation='relu'),
                         Dropout(0.3),
                         Dense(32,activation='relu'),
                         Dense(1,activation='sigmoid')])


#Model compilation

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Model training

history=model.fit_generator(train_gen,epochs=100,validation_data=test_gen)

#Plotting graph

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.grid('on')

#Saving the model

model.save("cats-vs-dogs.h5") 

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
train_data_dir = "D:/Datasets/Covid19-dataset/train"
test_data_dir = "D:/Datasets/Covid19-dataset/test"
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load your image dataset (replace 'your_dataset_directory' with your actual dataset directory)
# The dataset should be organized into subdirectories, each representing a class
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    'D:/Datasets/Covid19-dataset/train',
    target_size=(224, 224),  # Adjust the target size based on your requirements
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'D:/Datasets/Covid19-dataset/test',
    target_size=(224, 224),  # Adjust the target size based on your requirements
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Build the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))  # Adjust num_classes based on the number of classes in your dataset

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=validation_generator,batch_size=50)

# Evaluate the model
# accuracy = model.evaluate(validation_generator)[1]
# print(f'Validation Accuracy: {accuracy}')

# Make predictions
# predictions = model.predict(test_data)  # Provide your test data for predictions

# You can further customize the model architecture, hyperparameters, and evaluation based on your specific needs.

# img_width, img_height = 224, 224
# batch_size = 32
# train_datagen = ImageDataGenerator(rescale=1.0/255.0)
# test_datagen = ImageDataGenerator(rescale=1.0/255.0)
# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical',
#     shuffle=False
# )
# test_generator = test_datagen.flow_from_directory(
#     test_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical',
#     shuffle=False
# )

# base_model = VGG16(weights='imagenet', include_top=False,input_shape=(img_width, img_height,3))
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
# predictions = Dense(3, activation='softmax')(x)
# model = Model(inputs=base_model.input, outputs=predictions)
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(
#     train_generator,
#     steps_per_epoch=len(train_generator),
#     validation_data=test_generator,
#     validation_steps=len(test_generator),
#     epochs=15
# )

# Save the model structure and weights to a .hdf5 file
model.save('covid_detection_model.h5')
print("Model saved")



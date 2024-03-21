from CNN_net import SimpleVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import utils_paths
import matplotlib.pyplot as plt
from cv2 import cv2
import numpy as np
import argparse
import random
import pickle

import os

# Read data and tags
print("------Start reading data ------")
data = []
labels = []

# Obtain the image data path for subsequent reading
imagePaths = sorted(list(utils_paths.list_images('./dataset')))
random.seed(42)
random.shuffle(imagePaths)

image_size = 256
# Read data through traversal
for imagePath in imagePaths:
    # Read image data
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (image_size, image_size))
    data.append(image)
    # Read tags
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split data set
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Convert tags to the one-hot encoding format
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Data enhancement processing
aug = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest")

# Establish a convolutional neural network
model = SimpleVGGNet.build(width=256, height=256, depth=3,classes=len(lb.classes_))

# Set initialization hyper-parameters

# Learning rate
INIT_LR = 0.01
# Epoch
# Setting at 5 here is to finish training as soon as possible. You can set it higher, such as 30
EPOCHS = 5
# Batch Size
BS = 32

# Use the loss function, and compile the model
print("------Start training network------")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# Train network model
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS
)


# Test
print("------Test network------")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# Draw the result curve
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('./output/cnn_plot.png')

# Save the model
print("------Save the model------")
model.save('./cnn.model.h5')
f = open('./cnn_lb.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()
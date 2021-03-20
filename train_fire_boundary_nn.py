""" train_fire_boundary_nn.py

Trains a simple, feed-forward neural network for binary classification
of images, which either contain a fire boundary or do not.
"""

# Use matplotlib for plotting loss and accuracy and saving figures to output
import matplotlib
matplotlib.use("Agg")

# Use scikitlearn for partitioning data and tensorflow/keras for modeling and training the 
# neural network
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

#initialize the lists to hold the data and corresponding labels
data = []
labels = []

# create a list consisting of the paths to each image in my images directory, and
# then randomly shuffle the list
# this code is idiosyncratic to the directories on my machine (ie, not generalized)
# adjust this section as necessary if testing
imagePaths = []
path = os.getcwd()
files = os.listdir(path)
for file in files:
        if os.path.isfile(file):
                imagePaths.append(file)
                
random.shuffle(imagePaths)
print(imagePaths)

for imagePath in imagePaths:
	#using opencv, load the image, resize it to 32x32px to standardize it for the nn,
	#and flatten it into a single vector. Then append the vector to our data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32)).flatten()
	data.append(image)
	
	#extract the class label from the image path and append to our labels list
	label = imagePath.split("_")[-2]
	labels.append(label)
	
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)
	
# convert the labels from integers to vectors
trainY = to_categorical(trainY)
testY = to_categorical(testY)

# define the neural network architecture. We will use 1 input layer, 2 hidden layers, and 
# an output layer. The input layer will have 3072 nodes (one for each pixel). The output layer
# will have only one node, since this is a binary classification
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(1, activation="sigmoid"))

# initial learning rate and epochs to train for (learning rate will be adjusted based on 
# performance
INIT_LR = 0.01
EPOCHS = 80

# compile using stochastic gradient descent and binary cross-entropy loss
opt = SGD(lr=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
	
# train the neural network
H = model.fit(x=trainX, y=trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=32)

# after the model finishes training, evaluate it using the test set
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1)))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("output/plot")

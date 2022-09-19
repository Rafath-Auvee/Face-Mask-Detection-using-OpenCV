import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix

for dirpath,dirnames,filenames in os.walk("New Masks Dataset"):
    print(f"there are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def view_image(target_dir, target_class):
    target_folder = target_dir + target_class
    random_image = random.sample(os.listdir(target_folder), 1)
    print(random_image)
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")
    print(f"image shape {img.shape}")
    return img
img = view_image("New Masks Dataset/Train/","Non Mask")
img = view_image("New Masks Dataset/Train/","Mask")
data=[]
labels=[]
no_mask=os.listdir("New Masks Dataset/Train/Non Mask/")
for a in no_mask:
    image = cv2.imread("New Masks Dataset/Train/Non Mask/"+a,)
    image = cv2.resize(image, (224, 224))
    data.append(image)
    labels.append(0)
no_mask=os.listdir("New Masks Dataset/Test/Non Mask/")
for a in no_mask:
    image = cv2.imread("New Masks Dataset/Test/Non Mask/"+a,)
    image = cv2.resize(image, (224, 224))
    data.append(image)
    labels.append(0)

mask=os.listdir("New Masks Dataset/Train/Mask/")
for a in mask:
    image = cv2.imread("New Masks Dataset/Train/Mask/"+a,)
    image = cv2.resize(image, (224, 224))
    data.append(image)
    labels.append(1)

mask=os.listdir("New Masks Dataset/Test/Mask/")

for a in mask:
    image = cv2.imread("New Masks Dataset/Test/Mask/"+a,)
    image = cv2.resize(image, (224, 224))
    data.append(image)
    labels.append(1)


data = np.array(data) / 255.0
labels = np.array(labels)

print(data.shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42,shuffle=True,
                                                    stratify = labels)

base_model = tf.keras.applications.MobileNet(input_shape=[224, 224, 3], weights="imagenet", include_top=False)

base_model.trainable = False

# for layer in base_model.layers[30:]:
#   layer.trainable = False


model = Flatten()(base_model.output)
model = Dense(units=256, activation="relu")(model)
model = Dense(units=64, activation="relu")(model)
prediction_layer = Dense(units=1, activation="sigmoid")(model)

model = Model(inputs=base_model.input, outputs=prediction_layer)
model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=15,validation_split= 0.1, batch_size=32)



predictions = model.predict(X_test)

predict=[]

for i in range(len(predictions)):
    if predictions[i][0]>0.5:
        predict.append(1)
    else:
        predict.append(0)

pd.DataFrame(confusion_matrix(y_test, predict), columns= ["No Mask", "Mask"], index = ["No Mask", "Mask"])


model_name = "mask_detection_best.h5"
tf.keras.models.save_model(model, model_name)
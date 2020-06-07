import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras.regularizers import l2
import os
<<<<<<< HEAD
from tensorflow.keras.callbacks import EarlyStopping
=======
>>>>>>> 1a1ab11b023608024be6ef3a98a461452ccbe448
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# command line argument"
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
mode = ap.parse_args().mode


# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
<<<<<<< HEAD
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
=======
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
>>>>>>> 1a1ab11b023608024be6ef3a98a461452ccbe448
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
<<<<<<< HEAD
    fig.savefig('../imgs/facial_correction_plot.png')
=======
    fig.savefig('plot.png')
>>>>>>> 1a1ab11b023608024be6ef3a98a461452ccbe448
    plt.show()

# Define data generators
train_dir = 'data_af_preprocessing/Facial_correction_data/train'
val_dir = 'data_af_preprocessing/Facial_correction_data/test'
#PARAMETER
num_train = 28709   #So anh train
num_val = 7178  #So anh load
batch_size = 64
num_epoch = 50

<<<<<<< HEAD

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
=======
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
>>>>>>> 1a1ab11b023608024be6ef3a98a461452ccbe448

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# Create the model
model = Sequential()

<<<<<<< HEAD
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1),kernel_regularizer=l2(0.001), bias_regularizer=l2(0.1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',kernel_regularizer=l2(0.001), bias_regularizer=l2(0.1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',kernel_regularizer=l2(0.001), bias_regularizer=l2(0.1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',kernel_regularizer=l2(0.001), bias_regularizer=l2(0.1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
=======
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', input_shape=(48,48,1),kernel_regularizer=l2(0.001), bias_regularizer=l2(0.1)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(5, 5), activation='relu',kernel_regularizer=l2(0.001), bias_regularizer=l2(0.1)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(128, kernel_size=(4, 4), activation='relu',kernel_regularizer=l2(0.001), bias_regularizer=l2(0.1)))
model.add(Flatten())
model.add(Dense(3072, activation='relu', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.1)))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.1)))
>>>>>>> 1a1ab11b023608024be6ef3a98a461452ccbe448

# If you want to train the same model or try other models, go for this
if mode == "train":
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
    model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=int(num_train // batch_size),
            epochs=num_epoch,
            validation_data=validation_generator,
<<<<<<< HEAD
            validation_steps=int(num_val // batch_size),callbacks=[es])
=======
            validation_steps=int(num_val // batch_size))
>>>>>>> 1a1ab11b023608024be6ef3a98a461452ccbe448
    plot_model_history(model_info)

    model.save_weights('Model/CNN_Facial_correction_weight.h5')
    model.save('Model/CNN_Facial_correction_model.h5')
    #Ket qua train, val
    model.evaluate(train_generator)
    model.evaluate(validation_generator)
    # emotions will be displayed on your face from the webcam feed
elif mode == "display":
    model.load_weights('Model/CNN_Facial_correction_weight.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

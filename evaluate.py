import os
import cv2
import time
import numpy as np
import pyttsx3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from helper_functions import video_to_np_Array

actions_20 = np.array([
    "Hello", "House", "Father", "Friday", "Good afternoon", "Good evening", "Good Morning", 
    "Green", "Grey", "Actor", "Fan", "How are you", "Parent", "Park", "Patient", 
    "Pink", "quiet", "rich", "sad", "Thank you"
])

actions_3 = np.array([
    "Hello", "How are you", "Thank you"
])

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def initialize_model_20():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(45, 258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(256, return_sequences=True, activation="relu"))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions_20.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights(r"new_model.h5")
    return model

def initialize_model_3():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(45, 258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(256, return_sequences=True, activation="relu"))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions_3.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights(r"170-0.83.hdf5")
    return model

model_20 = initialize_model_20()
model_3 = initialize_model_3()

try:
    os.mkdir("input-video")
except FileExistsError:
    pass

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Camera is running...")

recording = False
start_time = None
selected_model = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, "Press 's' to start", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    if recording:
        cv2.putText(frame, "Recording... Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Sign Language to Text/Speech Translation", frame)

    key = cv2.waitKey(1) & 0xFF

    if not recording and key == ord('s'):
        selected_model = "3-class"
        recording = True
        start_time = time.time()
        print("Recording...")
        
        out = cv2.VideoWriter('input-video\\input.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 10, (width, height))

    elif not recording and key == ord('d'):
        selected_model = "20-class"
        recording = True
        start_time = time.time()
        print("Recording started...")
        
        out = cv2.VideoWriter('input-video\\input.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 10, (width, height))

    if recording:
        out.write(frame)
        if time.time() - start_time > 5:
            recording = False
            print("Recording ended")
            out.release()

            out_np_array = video_to_np_Array("input-video\\input.mp4", remove_input=False)
            prediction = None
            predicted_word = None

            if selected_model == "3-class":
                prediction = model_3.predict(np.expand_dims(out_np_array, axis=0))
                arg_pred = np.argmax(prediction, axis=1)
                predicted_word = actions_3[arg_pred[0]]
            elif selected_model == "20-class":
                prediction = model_20.predict(np.expand_dims(out_np_array, axis=0))
                arg_pred = np.argmax(prediction, axis=1)
                predicted_word = actions_20[arg_pred[0]]

            if predicted_word:
                print(f"Predicted Word: {predicted_word}")
                cv2.putText(frame, f"Predicted Word: {predicted_word}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                engine.say(predicted_word)
                engine.runAndWait()
                cv2.imshow("Sign Language to Text/Speech Translation", frame)
                cv2.waitKey(2000)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
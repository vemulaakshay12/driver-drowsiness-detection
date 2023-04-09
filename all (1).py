import numpy as np
import cv2
import pickle
import pyttsx3
import dlib
from scipy.spatial import distance

engine = pyttsx3.init('sapi5')

def speak(audio):
        engine.say(audio)
        engine.runAndWait()

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('trainner.yml')

label ={"person name": 1}
with open('label.pickle', 'rb') as f:
    label = pickle.load(f)
    label = {v:k for k,v in label.items()}

    cap = cv2.VideoCapture(0)

count = 1

while  True:
    rect, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x,y,w,h) in faces:

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 60:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = label[id_]
            color = (0,0,255)
            stroke = 2
            if count ==1:
                speak('HELLO ' + name)
                count +=1


            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = hog_face_detector(gray)
            for face in faces:

                face_landmarks = dlib_facelandmark(gray, face)
                leftEye = []
                rightEye = []

                for n in range(36,42):
                    x = face_landmarks.part(n).x
                    y = face_landmarks.part(n).y
                    leftEye.append((x,y))
                    next_point = n+1
                    if n == 41:
                        next_point = 36
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

                for n in range(42,48):
                    x = face_landmarks.part(n).x
                    y = face_landmarks.part(n).y
                    rightEye.append((x,y))
                    next_point = n+1
                    if n == 47:
                        next_point = 42
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

                left_ear = calculate_EAR(leftEye)
                right_ear = calculate_EAR(rightEye)

                EAR = (left_ear+right_ear)/2
                EAR = round(EAR,2)
                if EAR < 0.25:
                    speak('DROWSY')
                    speak('Are you Sleepy?')

        color = (0, 255, 0)
        stroke = 2
        end_chord_x = x + w
        end_chord_y = x + h
        cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

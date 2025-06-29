import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array 

model = tf.keras.models.load_model("d:\\Python Projects\\Projects\\Emotion Detection Live\\ResNet50_Transfer_Learning (1).keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_classifier = cv2.CascadeClassifier("D:\\Python Projects\\Projects\\Emotion Detection Materials\\Emotion Detection Materials\\Emotion Detection Deployement\\haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

cv2.namedWindow("Emotion Detector", cv2.WINDOW_NORMAL)  


while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
       
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (224, 224))
        face = face.astype("float32") / 255.0
        face_rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        face_rgb = img_to_array(face_rgb)
        face_rgb = np.expand_dims(face_rgb, axis=0)
        prediction = model.predict(face_rgb)[0]
        emotion = emotion_labels[np.argmax(prediction)]
        
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

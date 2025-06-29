
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.models import load_model
model = load_model("D:\\Python Projects\\Projects\\Emotion Detection Live\\ResNet50_Transfer_Learning (1).keras") 

emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

def prepare_img(input_img):
  img = input_img.resize((224, 224))
  img =  img_to_array(img)
  img = np.expand_dims(img, axis = 0)
  img = img/255
  return img

def prediction(img):
  prepared_img = prepare_img(img)
  probability = model.predict(prepared_img)[0]
  prediction = emotion_labels[np.argmax(probability)]

  return prediction

interface = gr.Interface(
    fn=prediction,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Emotion Detection",
    description="Upload an image and see the predicted emotion."
)

interface.launch()
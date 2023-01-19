import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50

img_path = '/content/drive/MyDrive/Магистратура Политех/3 семестр/Распознавание образов/1. Классификация изображения с использованием Keras/toyota_supra.jpg'
img = keras.utils.load_img(img_path, target_size=(224,224))
plt.imshow(img)
plt.show()

def classify(img_path):
    img = image.image_utils.load_img(img_path, target_size=(224,224))
    model = resnet50.ResNet50()
    img_array = img_to_array(img)
    img_batch = np.expand_dims(img_array, axis = 0)
    img_preprocessed = preprocess_input(img_batch)
    prediction = model.predict(img_preprocessed)
    print(decode_predictions(prediction, top=3)[0])

classify('/content/drive/MyDrive/Магистратура Политех/3 семестр/Распознавание образов/1. Классификация изображения с использованием Keras/toyota_supra.jpg')
import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
model = load_model('detection_model.h5')
def preprocess_image(image_path):
    img_grid = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    img_grid = cv.resize(img_grid, (150,150))
    img_grid = np.array(img_grid) / 255.0
    img_array = np.expand_dims(img_grid, axis=0)
    return img_array
def predict_with_model(model, image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return prediction
def interpret_outputs(prediction):
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class

image_path = '../chest_xray/test/PNEUMONIA/person10_virus_35.jpeg'
# image_path = '../chest_xray/test/NORMAL/IM-0009-0001.jpeg'
prediction = predict_with_model(model, image_path)
print(prediction)
# model.summary()
# predicted_class = interpret_outputs(prediction)
# print("Predicted Class:", predicted_class)
# model.history()
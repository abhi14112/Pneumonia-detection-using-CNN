from PIL import ImageOps, Image
import numpy as np
import streamlit as st
from keras.models import load_model
import cv2 as cv
def preprocess_image(image):
    image = ImageOps.fit(image, (150, 150), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    image_array = cv.cvtColor(image_array, cv.COLOR_RGB2GRAY)  
    image_array = np.expand_dims(image_array, axis=0)   
    image_array = image_array / 255.0
    return image_array
def classify(image, model, class_names):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    class_idx=0
    if prediction[0][0]>0.65:
        class_idx = 1
    class_name = class_names[class_idx]
    conf_score = prediction[0][0]
    return class_name, conf_score

st.title('Pneumonia Detector')
st.header('Please upload X-ray image')
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])
model = load_model('detection_model.h5')
class_names = ["NORMAL", "PNEUMONIA"]
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)
    if st.button("Classify"):
        class_name, conf_score = classify(image, model,class_names)
        if class_name == "NORMAL":
            bg_color = "green"
        else:
            bg_color = "red"
        st.write(f'<div style="background-color:{bg_color}; padding: 10px; border-radius: 5px;"><h2 style="color:white;">{class_name}</h2></div>',unsafe_allow_html=True)
                #  <p style="color:white;">Score: {conf_score:.2f}</p>
        # st.write("## {}".format(class_name))

        # st.write("### Score: {:.2f}%".format(conf_score))

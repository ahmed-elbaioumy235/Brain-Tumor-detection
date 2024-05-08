import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

model = tf.keras.models.load_model('Brain_tumor.h5')  
class_name = [
"Glioma" , "Pituitary","No tumor","Meningioma"
]

def preprocess_image(image):
    
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized_image = cv2.resize(grayscale_image, (200, 200))
    rescaled_image = resized_image.astype('float32') / 255.0
    return rescaled_image

def predict(image):
    image = preprocess_image(image)
    img_batch = np.expand_dims(image, 0)
    pred = model.predict(img_batch)
    index = np.argmax(pred[0])
    y_pred = class_name[index]
    confidence = float(np.max(pred[0]))
    result = {
        'the sample is ': y_pred,
        'precentile ': f'{confidence}'
    }
    return result

def main():
    st.title('Image Classification')
    st.write('Upload an image for classification')

    file = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])

    if file is not None:
        image = np.array(Image.open(file))
        result = predict(image)

        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write('Prediction:', result['the sample is '])
        st.write('Confidence:', result['precentile '])
if __name__ == "__main__":
    main()
    

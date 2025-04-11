import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

# Load and preprocess the image
def model_predict(image_path):
    model = tf.keras.models.load_model(r"C:/Users/shrad/OneDrive/Desktop/Intership/CNN_plantdiseases_model.keras")
    img = cv2.imread(image_path)  # read the file and convert into array
    H,W,C = 224,224,3
    img = cv2.resize(img, (H, W)) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = np.array(img)
    img = img.astype("float32")  
    img = img / 255.0  # rescaling
    img = img.reshape(1,H,W,C) #reshaping 
    
    prediction = np.argmax(model.predict(img),axis=-1)[0]
    return prediction

#Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page",["HOME","DISEASE RECOGNITION"])

# Home Page
if(app_mode=="HOME"):
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture", unsafe_allow_html=True)

# Prediction Page
elif(app_mode=="DISEASE RECOGNITION"):
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    
    # Option to upload an image or use webcam for real-time detection
    st.subheader("Choose an Input Method:")
    input_method = st.selectbox("Select Input", ["Upload Image", "Use Webcam"])

    if input_method == "Upload Image":
        test_image = st.file_uploader("Choose an Image:")
        if test_image is not None:
            # Save the uploaded image to the working directory
            save_path = os.path.join(os.getcwd(), test_image.name)
            with open(save_path, "wb") as f:
                f.write(test_image.getbuffer())

            st.image(test_image, caption="Uploaded Image", width=300)

            if st.button("Predict"):
                st.snow()  # Add snow effect while predicting
                result_index = model_predict(save_path)
                class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                              'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                              'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
                              'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
                              'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                              'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
                              'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
                              'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                              'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 
                              'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
                              'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
                              'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
                st.success(f"Model Prediction: {class_name[result_index]}")
    
    elif input_method == "Use Webcam":
        # Capture image from webcam
        st.subheader("Capture Image using Webcam")
        camera_image = st.camera_input("Take a Picture")

        if camera_image is not None:
            # Save the captured image to a file
            save_path = os.path.join(os.getcwd(), "captured_image.jpg")
            with open(save_path, "wb") as f:
                f.write(camera_image.getbuffer())

            st.image(camera_image, caption="Captured Image", width=300)

            if st.button("Predict"):
                st.snow()  # Add snow effect while predicting
                result_index = model_predict(save_path)
                class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                              'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                              'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
                              'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
                              'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                              'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
                              'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
                              'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                              'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 
                              'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
                              'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
                              'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
                st.success(f"Model Prediction: {class_name[result_index]}")

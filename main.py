import streamlit as st
import tensorflow as tf
import numpy as np

import os
import gdown

def download_model():
    model_url = "https://drive.google.com/uc?id=1UP2dc_CaCEogKGw9aoq_vXTl7noIohQL"
    model_path = "trained_plant_model.keras"
    
    if not os.path.exists(model_path):
        with st.spinner("🔄 Downloading model..."):
            gdown.download(model_url, model_path, quiet=False, fuzzy=True)

def model_predict(test_img):
    download_model()
    model = tf.keras.models.load_model("trained_plant_model.keras")

    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(test_img, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = input_arr / 255.0  # Normalize
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(input_arr)

    # Logging shape for debugging
    st.write("🔍 Raw prediction output:", prediction)
    st.write("Prediction shape:", prediction.shape)

    # Ensure valid prediction
    if prediction is not None and prediction.size > 0:
        return prediction
    else:
        return None


# Class names from PlantVillage dataset
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Sidebar
st.sidebar.title("🌿 Plant Disease System")
st.sidebar.markdown("### Navigation")
app_mode = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📖 About", "🔬 Disease Recognition"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Info")
if app_mode == "🏠 Home":
    st.sidebar.info("Welcome! Upload plant images and detect diseases easily.")
elif app_mode == "📖 About":
    st.sidebar.info("Learn about the dataset and the project goals.")
elif app_mode == "🔬 Disease Recognition":
    st.sidebar.info("Upload leaf images and get disease predictions.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Contact")
st.sidebar.markdown("--")

# Main content
if app_mode == "🏠 Home":
    st.header("Plant Disease Recognition System")
    try:
        st.image("plant_img.jpg", use_container_width=True)
    except Exception as e:
        st.warning("Home page image not found. Please ensure 'plant_img.jpg' is in the repository.")
    st.markdown("""
# 🌿 Plant Disease Recognition System

Welcome to the Plant Disease Recognition System — your smart assistant for identifying plant health issues early and accurately.

Our goal is to empower farmers, researchers, and plant enthusiasts with an easy-to-use tool that uses AI to detect plant diseases from images. Upload a photo of a plant leaf, and our system will analyze it to identify any visible signs of disease — helping you take timely action to protect your crops.

---

### 🚀 How It Works

1. **Upload an Image:** Head over to the **Disease Recognition** page in the sidebar.
2. **AI-Powered Detection:** Our deep learning model processes the image to detect potential diseases.
3. **Instant Results:** View the predicted disease and get actionable recommendations — all in seconds.

---

### ✅ Why Use This Tool?

- 🔍 **High Accuracy:** Built using advanced convolutional neural networks trained on thousands of labeled images.
- 🧑‍🌾 **Farmer-Friendly:** Designed for simplicity — no tech background needed.
- ⚡ **Fast & Reliable:** Get quick, accurate results with minimal wait.

---

### 🌱 Get Started

Click on the **Disease Recognition** link in the sidebar to begin. Upload your image and see the AI in action!

---

### 👥 About This Project

Want to know more about how this system was built or meet the team behind it? Visit the **About** page.

---
""")
elif app_mode == "📖 About":
    st.header("About")
    st.markdown("""
### 📊 About the Dataset

This dataset is based on the PlantVillage dataset, available on [GitHub](https://github.com/spMohanty/PlantVillage-Dataset), enhanced with offline data augmentation.

It contains approximately **87,000 high-quality RGB images** of **healthy and diseased crop leaves**, spanning across **38 different classes**. The dataset is carefully organized to support deep learning model training and evaluation.

The images are divided in an **80/20 split** for training and validation, while maintaining the original folder structure for each class.

Additionally, a separate **test set of 33 images** has been created for predictionierig

---
### 📁 Dataset Structure

1. **Train** – 70,295 images  
2. **Validation** – 17,572 images  
3. **Test** – 33 images  

Each image belongs to one of the 38 categories representing a specific crop and disease condition.

---
""")
elif app_mode == "🔬 Disease Recognition":
    st.header("Disease Recognition")
    test_img = st.file_uploader("Select an image:", type=["jpg", "jpeg", "png"])
    
    if test_img is not None:
        if st.button("Show Image"):
            st.image(test_img, use_container_width=True)
        
        if st.button("Predict"):
            st.write("Our Prediction")
            prediction = model_predict(test_img)

            if prediction is not None:
                # Check shape before indexing
                if prediction.ndim == 1:
                    result_index = np.argmax(prediction)
                    confidence = np.max(prediction) * 100
                elif prediction.ndim == 2:
                    result_index = np.argmax(prediction[0])
                    confidence = np.max(prediction[0]) * 100
                else:
                    st.error("Unexpected prediction format.")
                    st.stop()

                st.success(f"Model predicts: **{class_names[result_index]}** (Confidence: {confidence:.2f}%)")
            else:
                st.error("Prediction failed. Please try another image.")

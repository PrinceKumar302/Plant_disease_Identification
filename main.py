import streamlit as st
import tensorflow as tf
import numpy as np
import os
st.set_page_config(
    page_title="Plant Disease Detector 🌱",
    layout="centered",
)

#Tensorflow Model Prediction

def model_prediction(test_image):
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "trained_model.keras")
    model = tf.keras.models.load_model(model_path)

    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Home Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
""")

#About Page
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
    #### Content
    1. Train (70295 images)
    2. Valid (17572 image)
    3. Test (33 images)
    #### Our Team
    1. Prince Kumar Bhagat
    2. Pol Rohan Nitin
    3. Sumit Raj                        
""")
    
#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,use_column_width=True)
    #Predict Button
    if(st.button("Predict")):
        with st.spinner("Please Wait.."):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            #Define Class
            class_name = ['Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy']
            prediction_label = class_name[result_index]
            st.success(f"Model is Predicting it's a **{prediction_label}**")

# Disease treatment information (partial sample)
            disease_info = {
    "Apple___Apple_scab": "🛠️ Treatment: Remove infected leaves and use scab-resistant varieties. Apply fungicides early in the season.",
    "Apple___Black_rot": "🛠️ Treatment: Prune affected branches, apply fungicides, and remove fallen fruit.",
    "Apple___Cedar_apple_rust": "🛠️ Treatment: Use resistant varieties and fungicide sprays during early growth.",
    "Apple___healthy": "✅ Your apple plant appears to be healthy. Keep monitoring it regularly!",
    "Blueberry___healthy": "✅ Your blueberry plant appears to be healthy. Keep monitoring it regularly!",
    "Cherry_(including_sour)___Powdery_mildew": "🛠️ Treatment: Apply sulfur-based fungicides and ensure proper spacing between plants.",
    "Cherry_(including_sour)___healthy": "✅ Your cherry plant appears to be healthy. Keep monitoring it regularly!",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "🛠️ Treatment: Rotate crops and use fungicides if symptoms appear.",
    "Corn_(maize)___Common_rust_": "🛠️ Treatment: Apply fungicides and grow resistant hybrids.",
    "Corn_(maize)___Northern_Leaf_Blight": "🛠️ Treatment: Use resistant varieties and apply fungicides as needed.",
    "Corn_(maize)___healthy": "✅ Your corn plant appears to be healthy. Keep monitoring it regularly!",
    "Grape___Black_rot": "🛠️ Treatment: Remove and destroy infected grapes. Use fungicide sprays early in the season.",
    "Grape___Esca_(Black_Measles)": "🛠️ Treatment: Prune affected canes, avoid stress, and apply fungicides if necessary.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "🛠️ Treatment: Use protective fungicide sprays and ensure good air circulation.",
    "Grape___healthy": "✅ Your grapevine appears to be healthy. Keep monitoring it regularly!",
    "Orange___Haunglongbing_(Citrus_greening)": "🛠️ Treatment: Remove infected trees and control psyllid populations using insecticides.",
    "Peach___Bacterial_spot": "🛠️ Treatment: Use copper-based sprays and avoid overhead watering.",
    "Peach___healthy": "✅ Your peach tree appears to be healthy. Keep monitoring it regularly!",
    "Pepper,_bell___Bacterial_spot": "🛠️ Treatment: Use disease-free seeds and apply copper-based bactericides.",
    "Pepper,_bell___healthy": "✅ Your bell pepper plant appears to be healthy. Keep monitoring it regularly!",
    "Potato___Early_blight": "🛠️ Treatment: Apply fungicides and practice crop rotation.",
    "Potato___Late_blight": "🛠️ Treatment: Use resistant varieties and apply fungicides like mancozeb.",
    "Potato___healthy": "✅ Your potato plant appears to be healthy. Keep monitoring it regularly!",
    "Raspberry___healthy": "✅ Your raspberry plant appears to be healthy. Keep monitoring it regularly!",
    "Soybean___healthy": "✅ Your soybean crop appears to be healthy. Keep monitoring it regularly!",
    "Squash___Powdery_mildew": "🛠️ Treatment: Apply sulfur-based fungicides and maintain good air circulation.",
    "Strawberry___Leaf_scorch": "🛠️ Treatment: Remove infected leaves and use appropriate fungicides.",
    "Strawberry___healthy": "✅ Your strawberry plant appears to be healthy. Keep monitoring it regularly!",
    "Tomato___Bacterial_spot": "🛠️ Treatment: Use copper-based sprays and disease-free seeds.",
    "Tomato___Early_blight": "🛠️ Treatment: Use fungicides like chlorothalonil and remove affected foliage.",
    "Tomato___Late_blight": "🛠️ Treatment: Apply fungicides and destroy infected plants immediately.",
    "Tomato___Leaf_Mold": "🛠️ Treatment: Ensure good ventilation and apply fungicides like mancozeb.",
    "Tomato___Septoria_leaf_spot": "🛠️ Treatment: Remove affected leaves and apply fungicides early.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "🛠️ Treatment: Use miticides and maintain adequate humidity.",
    "Tomato___Target_Spot": "🛠️ Treatment: Apply preventive fungicides and remove infected leaves.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "🛠️ Treatment: Control whitefly populations and use resistant varieties.",
    "Tomato___Tomato_mosaic_virus": "🛠️ Treatment: Remove infected plants and sterilize tools regularly.",
    "Tomato___healthy": "✅ Your tomato plant appears to be healthy. Keep monitoring it regularly!"
}

            if prediction_label in disease_info:
                st.warning(disease_info[prediction_label])
            else:
                st.info("No specific treatment information found for this disease. Please consult an agricultural expert.")

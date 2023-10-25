import streamlit as st
import tensorflow as tf
import requests
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie

# --- MODEL LOADING ---
# model = tf.keras.models.load_model('model.h5')


# --- PAGE CONFIG  ---
st.set_page_config(page_title="Skin Disease Sorting", page_icon=":mask:", layout="wide")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_doctor = load_lottieurl("https://lottie.host/d0c230d5-814f-41bf-a814-86049c4ce776/ci59c1Cw0y.json")
lottie_analysis = load_lottieurl("https://lottie.host/29d717ad-813d-444a-87a9-8496d4d8c05c/4Tw6Tw5K7h.json")

# --- HEADER SECTION ---
with st.container():
    st.markdown("<h3 style='text-align: center; color: black;'>Welcome to</h3>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: black;'>Skin Disease Identification App</h1>", unsafe_allow_html=True)
    st_lottie(lottie_doctor, speed=1, height=300, key="doctor")
    st.markdown("<p style='text-align: center; font-size: 20px'>This application uses state-of-the-art machine learning to diagnose skin diseases.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 20px'>Simply upload an image and let the model diagnose it!</p>", unsafe_allow_html=True)



uploaded_file = st.file_uploader("Choose a skin lesion image...", type="jpg")

# --- ANALYSIS SECTION ---
if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
    # st.image(image, caption='Uploaded Image.', use_column_width=True)
    # st.write("")
    # st.write("Classifying...")

    # img_array = np.asarray(image.resize((224, 224)))  
    # img_array = img_array / 255.0
    # img_array = img_array[np.newaxis, ...]

    # predictions = model.predict(img_array)

    # Display the predictions

# --- RESULT SECTION ---

st_lottie(lottie_analysis, speed=1, height=300, key="initial")


# --- FOOTER SECTION ---
with st.container():
    st.write("---")
    st.header("Facts:")
    facts = [
        "Skin cancer is the most common cancer globally.",
        "There are mainly three types of skin cancer: basal cell carcinoma, squamous cell carcinoma, and melanoma.",
        "Melanoma is the deadliest form of skin cancer but is less common than the other types.",
        "Ultraviolet (UV) radiation from the sun is the main cause of skin cancer. Tanning lamps and beds are also sources of UV radiation.",
        "Early detection of skin cancer gives a patient a high chance of successful treatment.",
        "Regular self-exams and understanding the ABCDEs of melanoma can help in early detection. (Asymmetry, Border, Color, Diameter, Evolving)",
        "Using broad-spectrum (UVA/UVB) sunscreen and wearing protective clothing can reduce the risk of skin cancer.",
        "While skin cancer is common in many parts of the world, countries with predominantly fair-skinned populations and high UV exposure, such as Australia, have particularly high rates.",
        "Individuals who have had skin cancer once are at a higher risk of getting it again.",
        "People with weakened immune systems, such as organ transplant recipients, are at greater risk.",
        "A family history of skin cancer can increase one's risk."
    ]

    for fact in facts:
        st.markdown(f"- {fact}")

    st.write("Awareness and understanding of these facts can play a significant role in prevention and early detection. Regular check-ups with dermatologists are also crucial for those at higher risk.")

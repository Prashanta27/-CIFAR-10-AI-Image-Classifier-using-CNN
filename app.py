import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import pandas as pd
import time
import matplotlib.pyplot as plt


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="CIFAR-10 AI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* Main App */
.stApp {
    background: linear-gradient(to bottom right, #0f172a, #111827);
    color: white;
    font-family: 'Poppins', sans-serif;
}

/* BIG TITLE */
.custom-title {
    font-size: 75px !important;
    font-weight: 900 !important;
    text-align: center;
    color: #38bdf8 !important;
    margin-bottom: 10px;
    text-shadow: 0px 0px 25px rgba(56,189,248,0.6);
}

/* Subtitle */
.custom-subtitle {
    text-align: center;
    font-size: 28px !important;
    color: #cbd5e1 !important;
    margin-bottom: 40px;
}

/* Upload Box */
[data-testid="stFileUploader"] {
    background-color: #1e293b !important;
    border: 2px dashed #38bdf8 !important;
    border-radius: 20px !important;
    padding: 25px !important;
}

/* Upload text */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] p {
    color: white !important;
    font-size: 20px !important;
}

/* Upload Icon */
[data-testid="stFileUploader"] svg {
    fill: #38bdf8 !important;
    color: #38bdf8 !important;
    width: 55px !important;
    height: 55px !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white !important;
    font-size: 22px !important;
    border-radius: 14px;
    border: none;
    padding: 12px 20px;
    font-weight: bold;
}

/* General Text */
html, body, p, div, span, label {
    color: white !important;
    font-size: 20px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827 !important;
}

section[data-testid="stSidebar"] * {
    color: white !important;
    font-size: 18px !important;
}

</style>
""", unsafe_allow_html=True)
# =========================
# CUSTOM CSS (Industry Level UI)
# =========================
st.markdown("""
<style>

.main {
    background: linear-gradient(to bottom right, #0f172a, #111827);
    color: white;
}

.stApp {
    background: linear-gradient(to bottom right, #0f172a, #111827);
}

h1, h2, h3, h4 {
    color: white !important;
}

.custom-title {
    font-size: 30px;
    font-weight: 800;
    text-align: center;
    color: white;
    margin-bottom: 10px;
}

.custom-subtitle {
    text-align: center;
    font-size: 20px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.card {
    background: rgba(255,255,255,0.06);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
}

.prediction-box {
    padding: 18px;
    border-radius: 15px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    font-size: 30px;
    font-weight: bold;
    text-align: center;
    margin-top: 20px;
}

.footer {
    text-align:center;
    margin-top:40px;
    color:#94a3b8;
}

[data-testid="stSidebar"] {
    background: #0f172a;
}

</style>
""", unsafe_allow_html=True)

# =========================
# CIFAR-10 CLASSES
# =========================
CLASSES = [
    'Airplane ✈️',
    'Automobile 🚗',
    'Bird 🐦',
    'Cat 🐱',
    'Deer 🦌',
    'Dog 🐶',
    'Frog 🐸',
    'Horse 🐴',
    'Ship 🚢',
    'Truck 🚚'
]

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "final_cifar10_cnn.keras"

@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        return None

model = load_model()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("🧠 CNN Dashboard")
    
    st.markdown("---")
    
    st.markdown("""
    ### 📌 About Project
    
    This AI model is trained on the CIFAR-10 dataset using Convolutional Neural Networks (CNN).
    
    ### 🔍 Detectable Classes
    - Airplane
    - Car
    - Bird
    - Cat
    - Deer
    - Dog
    - Frog
    - Horse
    - Ship
    - Truck
    """)
    
    st.markdown("---")
    
    if model is not None:
        st.success("✅ Model Loaded Successfully")
    else:
        st.error("❌ Model Not Found")

# =========================
# HEADER
# =========================
st.markdown('<div class="custom-title">🚀 CIFAR-10 AI Image Classifier</div>', unsafe_allow_html=True)



# =========================
# MAIN LAYOUT
# =========================
col1, col2 = st.columns([1,1], gap="large")

# =========================
# IMAGE UPLOAD SECTION
# =========================
with col1:
    st.markdown("""
<style>

/* Upload Section Title */
.upload-title {
    color: #38bdf8;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 15px;
}

/* Upload Box */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border: 3px dashed #38bdf8;
    border-radius: 22px;
    padding: 30px;
    text-align: center;
}

/* Upload Icon */
[data-testid="stFileUploader"] svg {
    fill: #38bdf8 !important;
    color: #38bdf8 !important;
    width: 70px !important;
    height: 70px !important;
}

/* Upload Text */
[data-testid="stFileUploader"] * {
    color: white !important;
    font-size: 22px !important;
}

/* Browse Files Button */
[data-testid="stBaseButton-secondary"] {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-size: 20px !important;
    font-weight: bold !important;
    padding: 10px 20px !important;
}

/* Uploaded Image Caption */
[data-testid="stImageCaption"] {
    color: #cbd5e1 !important;
    font-size: 20px !important;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.subheader("📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREDICTION SECTION
# =========================
with col2:
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.subheader("🤖 AI Prediction")
    
    if uploaded_file:
        
        if st.button("🔍 Analyze Image", use_container_width=True):
            
            if model is None:
                st.error("Model file not found.")
            
            else:
                with st.spinner("AI is analyzing the image..."):
                    
                    time.sleep(1)

                    # Preprocessing
                    img = image.resize((32, 32))
                    img_array = np.array(img)

                    # RGB Handling
                    if len(img_array.shape) == 2:
                        img_array = np.stack([img_array]*3, axis=2)

                    elif img_array.shape[2] == 4:
                        img_array = img_array[:, :, :3]

                    # Normalize
                    img_array = img_array.astype("float32") / 255.0

                    # Batch Dimension
                    img_array = np.expand_dims(img_array, axis=0)

                    # Prediction
                    prediction = model.predict(img_array, verbose=0)

                    # Softmax
                    confidence_scores = tf.nn.softmax(prediction[0]).numpy()

                    # Top Prediction
                    predicted_index = np.argmax(confidence_scores)
                    predicted_class = CLASSES[predicted_index]
                    confidence = confidence_scores[predicted_index] * 100

                    # Prediction Box
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                        🎯 Prediction: {predicted_class}<br>
                        📊 Confidence: {confidence:.2f}%
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown("### 📈 Confidence Scores")

                    # Dataframe
                    df = pd.DataFrame({
                        "Class": CLASSES,
                        "Confidence": confidence_scores
                    })

                    df = df.sort_values(
                        by="Confidence",
                        ascending=False
                    )


                    fig, ax = plt.subplots(figsize=(8,4))

                    ax.bar(df["Class"], df["Confidence"])

                    plt.xticks(rotation=45)

                    st.pyplot(fig)

                    # Table
                    st.markdown("""
                    <style>

                    table {
                    font-size: 20px !important;
                    }

                    thead tr th {
                    font-size: 22px !important;
                    font-weight: bold !important;
                    }

                    tbody tr td {
                    font-size: 20px !important;
                    }

                    </style>
                    """, unsafe_allow_html=True)
                    df["Confidence"] = (df["Confidence"] * 100).round(2).astype(str) + "%"
                    
                    st.table(df)

    else:
        st.info("Upload an image to start prediction.")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
Built with ❤️ . What Are You Feeling?
</div>
""", unsafe_allow_html=True)
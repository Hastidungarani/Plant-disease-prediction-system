import os
import json
import base64
import datetime
import io
import requests
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import cv2

import numpy as np
import tensorflow as tf
import streamlit as st

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

from googletrans import Translator



# Page configuration
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="wide")

# Language support
translator = Translator()

lang_map = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Tamil": "ta",
    "Telugu": "te",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Urdu": "ur"
}


# Sidebar language selector
st.sidebar.markdown('<div class="section-title">🌐 Language</div>', unsafe_allow_html=True)
selected_lang_name = st.sidebar.selectbox("Choose your preferred language", list(lang_map.keys()))
selected_lang = lang_map[selected_lang_name]

# Translate helper
def t(text):
    if selected_lang != "en":
        try:
            return translator.translate(text, dest=selected_lang).text
        except:
            return text
    return text

# Encode background image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return f"data:image/jpg;base64,{base64.b64encode(img_file.read()).decode()}"

# Set background image
bg_img = get_base64_image("leaves.jpg")

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []


# Custom CSS styling with updated font color
# Updated CSS for new background and font contrast
# Updated CSS for clear readability on leafy background
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("{bg_img}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: #ffffff;
        font-size: 18px;
    }}

    html, body {{
        font-family: 'Segoe UI', sans-serif;
        padding: 0;
        margin: 0;
        color: #ffffff;
    }}

    .header {{
        background-color: rgba(0, 51, 0, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }}

    .content-section {{
        background-color: rgba(0, 0, 0, 0.6);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
    }}

    .section-title {{
        color: #c8facc;
        font-size: 1.6rem;
        font-weight: bold;
        margin-bottom: 1rem;
        border-bottom: 2px solid #81c784;
        padding-bottom: 0.3rem;
    }}

    .stButton > button {{
        background-color: #4CAF50;
        color: #ffffff;
        font-weight: 600;
        border-radius: 25px;
        padding: 0.6rem 2rem;
        border: none;
        transition: 0.3s ease;
    }}
    .stButton > button:hover {{
        background-color: #388E3C;
        transform: scale(1.03);
    }}

    .result-box {{
        padding: 1.2rem;
        border-radius: 12px;
        margin-top: 1rem;
        color: #ffffff;
    }}

    .healthy-result {{
        background-color: rgba(76, 175, 80, 0.7);
        border-left: 5px solid #ffffff;
    }}
    .disease-result {{
        background-color: rgba(255, 152, 0, 0.7);
        border-left: 5px solid #ffffff;
    }}

    .step-box {{
        background-color: rgba(255, 255, 255, 0.15);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffffff;
        color: #ffffff;
    }}

    .image-container {{
        padding: 0.5rem;
        background-color: rgba(0,0,0,0.5);
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
    }}

    .footer {{
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        background-color: rgba(0, 51, 0, 0.6);
        border-radius: 10px;
        color: #ffffff;
    }}
    </style>
""", unsafe_allow_html=True)

# Load model and classes
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("project2.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_class_indices():
    try:
        with open("class_indices.json") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading class indices: {str(e)}")
        return None

@st.cache_data
def load_disease_info():
    """Load detailed information about plant diseases"""
    return {
        "leaf_blight": {
            "description": "Leaf blight is a common disease characterized by rapid browning and death of leaf tissue, often starting at the margins.",
            "causes": ["Fungal pathogens (Alternaria, Phytophthora)", "Bacterial infections", "Environmental stress"],
            "symptoms": ["Brown/black lesions on leaves", "Yellowing around lesions", "Rapid wilting", "Leaf death"],
            "treatments": ["Remove infected leaves", "Apply copper-based fungicides", "Improve drainage", "Avoid overhead irrigation"],
            "prevention": ["Crop rotation", "Adequate plant spacing", "Clean gardening tools", "Resistant varieties"]
        },
        "leaf_spot": {
            "description": "Leaf spot diseases cause spots on foliage. The spots will vary in size and color depending on the plant affected, the specific organism involved, and the stage of development.",
            "causes": ["Fungal infection", "Bacterial pathogens", "Environmental factors"],
            "symptoms": ["Circular spots on leaves", "Dark borders around spots", "Yellowing leaves", "Leaf drop"],
            "treatments": ["Remove infected parts", "Apply fungicide", "Improve air circulation", "Proper watering"],
            "prevention": ["Avoid wetting foliage", "Space plants properly", "Sanitize garden tools", "Regular inspection"]
        },
        "powdery_mildew": {
            "description": "Powdery mildew is a fungal disease that affects a wide range of plants, appearing as a white to gray powdery coating on leaf surfaces.",
            "causes": ["Fungal spores", "High humidity with poor air circulation", "Moderate temperatures (60-80°F)"],
            "symptoms": ["White powdery spots on leaves and stems", "Yellowing leaves", "Distorted growth", "Premature leaf drop"],
            "treatments": ["Apply fungicides with sulfur", "Potassium bicarbonate sprays", "Neem oil applications", "Milk spray (1:10 milk to water)"],
            "prevention": ["Improve air circulation", "Avoid overhead watering", "Plant resistant varieties", "Space plants adequately"]
        },
        "bacterial_leaf_spot": {
            "description": "Bacterial leaf spot is a disease that produces dark, water-soaked lesions on leaves, which can spread to stems and fruit.",
            "causes": ["Bacterial pathogens (Xanthomonas, Pseudomonas)", "Warm, wet conditions", "Splashing water"],
            "symptoms": ["Water-soaked spots", "Yellow halos around lesions", "Angular spots", "Spots may turn black or brown"],
            "treatments": ["Remove infected plant material", "Copper-based bactericides", "Avoid overhead watering", "Improve air circulation"],
            "prevention": ["Pathogen-free seeds", "Crop rotation", "Avoid working with wet plants", "Clean tools between plants"]
        },
        "rust": {
            "description": "Rust is a fungal disease that appears as rusty spots or pustules on leaves and stems, reducing plant vigor and yield.",
            "causes": ["Fungal pathogens (Puccinia, Uromyces)", "High humidity", "Moderate temperatures"],
            "symptoms": ["Orange-brown pustules on lower leaf surfaces", "Yellow spots on upper leaf surfaces", "Distorted growth", "Premature defoliation"],
            "treatments": ["Remove infected plant parts", "Apply sulfur-based fungicides", "Use neem oil", "Potassium bicarbonate sprays"],
            "prevention": ["Adequate plant spacing", "Morning watering", "Resistant varieties", "Good sanitation"]
        },
        "early_blight": {
            "description": "Early blight is a fungal disease causing dark lesions with concentric rings on lower, older leaves first.",
            "causes": ["Alternaria fungus", "Warm, humid conditions", "Poor air circulation"],
            "symptoms": ["Dark brown spots with concentric rings", "Yellow areas around spots", "Older leaves affected first", "Spots may merge"],
            "treatments": ["Remove infected leaves", "Copper-based fungicides", "Organic fungicides with Bacillus subtilis", "Proper spacing"],
            "prevention": ["Mulch around plants", "Crop rotation", "Stake plants", "Water at soil level"]
        },
        "late_blight": {
            "description": "Late blight is a devastating disease that can kill plants within days, most infamously causing the Irish Potato Famine.",
            "causes": ["Phytophthora infestans fungus", "Cool, wet weather", "High humidity"],
            "symptoms": ["Water-soaked spots", "White fuzzy growth on undersides", "Rapid browning and wilting", "Can affect all plant parts"],
            "treatments": ["Remove and destroy infected plants", "Copper-based fungicides (preventive)", "Maintain dry foliage", "Improve drainage"],
            "prevention": ["Plant resistant varieties", "Wide spacing", "Avoid overhead watering", "Plant in well-drained soil"]
        }
    }

model = load_model()
class_indices = load_class_indices()
disease_info = load_disease_info()

@st.cache_data
def load_disease_info():
    """Load detailed information about plant diseases"""
    return {
        "leaf_blight": {
            "description": "Leaf blight is a common disease characterized by rapid browning and death of leaf tissue, often starting at the margins.",
            "causes": ["Fungal pathogens (Alternaria, Phytophthora)", "Bacterial infections", "Environmental stress"],
            "symptoms": ["Brown/black lesions on leaves", "Yellowing around lesions", "Rapid wilting", "Leaf death"],
            "treatments": ["Remove infected leaves", "Apply copper-based fungicides", "Improve drainage", "Avoid overhead irrigation"],
            "prevention": ["Crop rotation", "Adequate plant spacing", "Clean gardening tools", "Resistant varieties"],
            "progress_timeline": ["Day 1-3: Isolate plant and remove infected leaves", 
                                 "Day 4-7: Apply appropriate fungicide treatment", 
                                 "Day 8-14: Monitor for new infections and continue treatment if needed",
                                 "Day 15-30: Focus on prevention and improving plant health"]
        },
        "leaf_spot": {
            "description": "Leaf spot diseases cause spots on foliage. The spots will vary in size and color depending on the plant affected, the specific organism involved, and the stage of development.",
            "causes": ["Fungal infection", "Bacterial pathogens", "Environmental factors"],
            "symptoms": ["Circular spots on leaves", "Dark borders around spots", "Yellowing leaves", "Leaf drop"],
            "treatments": ["Remove infected parts", "Apply fungicide", "Improve air circulation", "Proper watering"],
            "prevention": ["Avoid wetting foliage", "Space plants properly", "Sanitize garden tools", "Regular inspection"],
            "progress_timeline": ["Day 1-2: Remove and destroy infected leaves", 
                               "Day 3-5: Apply appropriate fungicide treatment", 
                               "Day 6-10: Ensure good air circulation around plants",
                               "Day 11-21: Monitor for new infections, apply second treatment if needed"]
        },
        "powdery_mildew": {
            "description": "Powdery mildew is a fungal disease that affects a wide range of plants, appearing as a white to gray powdery coating on leaf surfaces.",
            "causes": ["Fungal spores", "High humidity with poor air circulation", "Moderate temperatures (60-80°F)"],
            "symptoms": ["White powdery spots on leaves and stems", "Yellowing leaves", "Distorted growth", "Premature leaf drop"],
            "treatments": ["Apply fungicides with sulfur", "Potassium bicarbonate sprays", "Neem oil applications", "Milk spray (1:10 milk to water)"],
            "prevention": ["Improve air circulation", "Avoid overhead watering", "Plant resistant varieties", "Space plants adequately"],
            "progress_timeline": ["Day 1: Remove severely affected leaves and isolate plant", 
                               "Day 2-3: Apply organic fungicide like neem oil or milk spray", 
                               "Day 4-7: Improve air circulation and reduce humidity around plant",
                               "Day 8-14: Apply second treatment if needed",
                               "Day 15-28: Monitor for recurrence and maintain preventive measures"]
        },
        "bacterial_leaf_spot": {
            "description": "Bacterial leaf spot is a disease that produces dark, water-soaked lesions on leaves, which can spread to stems and fruit.",
            "causes": ["Bacterial pathogens (Xanthomonas, Pseudomonas)", "Warm, wet conditions", "Splashing water"],
            "symptoms": ["Water-soaked spots", "Yellow halos around lesions", "Angular spots", "Spots may turn black or brown"],
            "treatments": ["Remove infected plant material", "Copper-based bactericides", "Avoid overhead watering", "Improve air circulation"],
            "prevention": ["Pathogen-free seeds", "Crop rotation", "Avoid working with wet plants", "Clean tools between plants"],
            "progress_timeline": ["Day 1-2: Remove and destroy all infected leaves", 
                               "Day 3-4: Apply copper-based bactericide", 
                               "Day 5-10: Ensure plants receive water at soil level only",
                               "Day 11-14: Apply second treatment if new spots appear",
                               "Day 15-30: Monitor carefully and improve growing conditions"]
        },
        "rust": {
            "description": "Rust is a fungal disease that appears as rusty spots or pustules on leaves and stems, reducing plant vigor and yield.",
            "causes": ["Fungal pathogens (Puccinia, Uromyces)", "High humidity", "Moderate temperatures"],
            "symptoms": ["Orange-brown pustules on lower leaf surfaces", "Yellow spots on upper leaf surfaces", "Distorted growth", "Premature defoliation"],
            "treatments": ["Remove infected plant parts", "Apply sulfur-based fungicides", "Use neem oil", "Potassium bicarbonate sprays"],
            "prevention": ["Adequate plant spacing", "Morning watering", "Resistant varieties", "Good sanitation"],
            "progress_timeline": ["Day 1: Remove all infected leaves and stems", 
                               "Day 2-3: Apply sulfur-based fungicide", 
                               "Day 4-7: Increase air circulation around plants",
                               "Day 8-14: Monitor for new infections, reapply fungicide if needed",
                               "Day 15-30: Continue monitoring and maintain proper sanitation"]
        },
        "early_blight": {
            "description": "Early blight is a fungal disease causing dark lesions with concentric rings on lower, older leaves first.",
            "causes": ["Alternaria fungus", "Warm, humid conditions", "Poor air circulation"],
            "symptoms": ["Dark brown spots with concentric rings", "Yellow areas around spots", "Older leaves affected first", "Spots may merge"],
            "treatments": ["Remove infected leaves", "Copper-based fungicides", "Organic fungicides with Bacillus subtilis", "Proper spacing"],
            "prevention": ["Mulch around plants", "Crop rotation", "Stake plants", "Water at soil level"],
            "progress_timeline": ["Day 1-2: Remove all infected leaves, especially lower ones", 
                               "Day 3-4: Apply copper-based fungicide or organic alternative", 
                               "Day 5-7: Ensure proper mulching around plants",
                               "Day 8-14: Monitor spread and apply second treatment if needed",
                               "Day 15-30: Continue monitoring and maintain good air circulation"]
        },
        "late_blight": {
            "description": "Late blight is a devastating disease that can kill plants within days, most infamously causing the Irish Potato Famine.",
            "causes": ["Phytophthora infestans fungus", "Cool, wet weather", "High humidity"],
            "symptoms": ["Water-soaked spots", "White fuzzy growth on undersides", "Rapid browning and wilting", "Can affect all plant parts"],
            "treatments": ["Remove and destroy infected plants", "Copper-based fungicides (preventive)", "Maintain dry foliage", "Improve drainage"],
            "prevention": ["Plant resistant varieties", "Wide spacing", "Avoid overhead watering", "Plant in well-drained soil"],
            "progress_timeline": ["Day 1: Remove severely infected plants entirely and destroy", 
                               "Day 1-2: Apply protective fungicide to remaining plants", 
                               "Day 3-7: Ensure excellent air circulation and drainage",
                               "Day 8-14: Continue monitoring and reapply fungicide",
                               "Day 15-30: Maintain vigilance as this disease spreads rapidly"]
        }
    }

def get_seasonal_advice(season):
    """Get plant care advice based on the current season"""
    advice = {
        "Spring": {
            "disease_risks": [
                t("Early appearance of powdery mildew"),
                t("Seedling damping off"),
                t("Root rots in wet conditions")
            ],
            "care_tips": [
                t("Monitor new growth closely for early disease signs"),
                t("Avoid overwatering as temperatures are still variable"),
                t("Apply preventative fungicides before disease pressure builds"),
                t("Keep an eye on weather forecasts for frost warnings")
            ],
            "icon": "🌱"
        },
        "Summer": {
            "disease_risks": [
                t("Fungal diseases in humid conditions"),
                t("Bacterial spots accelerated by warm weather"),
                t("Spider mites and other pests")
            ],
            "care_tips": [
                t("Water early morning to reduce leaf wetness duration"),
                t("Increase spacing between plants for better airflow"),
                t("Monitor for heat stress which can weaken disease resistance"),
                t("Apply organic mulch to regulate soil temperature and moisture")
            ],
            "icon": "☀️"
        },
        "Fall": {
            "disease_risks": [
                t("Late blight in cool, wet conditions"),
                t("Leaf spot diseases spreading before dormancy"),
                t("Root diseases from excess moisture")
            ],
            "care_tips": [
                t("Remove and destroy fallen leaves to reduce disease carryover"),
                t("Reduce watering as temperatures cool"),
                t("Apply protective fungicides before winter if needed"),
                t("Prune out any diseased tissue before winter dormancy")
            ],
            "icon": "🍂"
        },
        "Winter": {
            "disease_risks": [
                t("Root rots in wet, cold soil"),
                t("Fungal diseases in greenhouse conditions"),
                t("Disease buildup in plant debris")
            ],
            "care_tips": [
                t("Ensure proper drainage around dormant plants"),
                t("For indoor plants, maintain good air circulation"),
                t("Sanitize tools and containers for spring planting"),
                t("Plan crop rotation to reduce disease pressure next season")
            ],
            "icon": "❄️"
        }
    }
    return advice[season]
# Function to enhance leaf image
def enhance_leaf_image(image, contrast=1.5, sharpen=True, denoise=True):
    """Enhance leaf image to improve detection accuracy"""
    img = image.copy()
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast)
    
    # Apply sharpening
    if sharpen:
        img = img.filter(ImageFilter.SHARPEN)
    
    # Convert to numpy for OpenCV processing
    if denoise:
        img_array = np.array(img)
        img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        img = Image.fromarray(img_array)
    
    return img

# Preprocess image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    if isinstance(image_path, Image.Image):
        img = image_path.convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Prediction function
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    confidence = float(predictions[0][predicted_class_index]) * 100
    return predicted_class_name, confidence, predictions[0]

# Function to generate PDF report
def generate_pdf_report(plant_type, condition, confidence, is_healthy, recommendations, normal_image=None, enhanced_image=None):
    """Generate a PDF report of the plant analysis without symptoms"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title Header
    c.setFont("Helvetica-Bold", 22)
    c.setFillColor(colors.darkgreen)
    c.drawString(50, height - 50, "🌿 Plant Disease Analysis Report")
    c.setStrokeColor(colors.darkgreen)
    c.setLineWidth(2)
    c.line(50, height - 55, width - 50, height - 55)

    # Date
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.black)
    c.drawString(50, height - 80, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}")

    # Section: Plant Info
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 120, "🌱 Plant Information")
    c.setFont("Helvetica", 12)
    c.drawString(70, height - 140, f"Plant Type: {plant_type}")
    c.drawString(70, height - 160, f"Condition: {condition}")
    c.drawString(70, height - 180, f"Confidence: {confidence:.2f}%")

    # Health status
    c.setFont("Helvetica-Bold", 12)
    if is_healthy:
        c.setFillColor(colors.green)
        status = "HEALTHY"
    else:
        c.setFillColor(colors.orange)
        status = "DISEASE DETECTED"
    c.rect(400, height - 140, 150, 30, fill=1)
    c.setFillColor(colors.white)
    c.drawString(430, height - 127, status)
    c.setFillColor(colors.black)

    # Section: Recommendations
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 220, "🛠️ Recommendations")
    c.setFont("Helvetica", 12)
    for i, rec in enumerate(recommendations):
        c.drawString(70, height - 240 - (i * 20), f"• {rec}")

    y_offset = height - 240 - (len(recommendations) * 20) - 20

    # Insert images (optional)
    if normal_image and enhanced_image:
        try:
            from reportlab.lib.utils import ImageReader
            image_io_normal = io.BytesIO()
            normal_image.save(image_io_normal, format='PNG')
            image_io_normal.seek(0)
            img_reader_normal = ImageReader(image_io_normal)
            c.drawImage(img_reader_normal, 50, y_offset - 150, width=200, height=150, preserveAspectRatio=True)

            image_io_enhanced = io.BytesIO()
            enhanced_image.save(image_io_enhanced, format='PNG')
            image_io_enhanced.seek(0)
            img_reader_enhanced = ImageReader(image_io_enhanced)
            c.drawImage(img_reader_enhanced, 300, y_offset - 150, width=200, height=150, preserveAspectRatio=True)

        except Exception as e:
            c.setFont("Helvetica", 10)
            c.setFillColor(colors.red)
            c.drawString(70, y_offset - 20, f"(Error displaying images in report: {str(e)})")

    # Footer
    c.setFont("Helvetica-Oblique", 10)
    c.setFillColor(colors.black)
    c.drawString(50, 50, "This report is generated by the Plant Disease Detector app.")
    c.drawString(50, 35, "For educational purposes only. Consult an expert for confirmation.")

    c.save()
    buffer.seek(0)
    return buffer

# UI COMPONENTS



# Seasonal advice widget
st.sidebar.markdown('<div class="section-title">Seasonal Advice</div>', unsafe_allow_html=True)
season = st.sidebar.selectbox("Select Season", ["Spring", "Summer", "Fall", "Winter"])
seasonal_advice = get_seasonal_advice(season)

st.sidebar.markdown(f"### Risks {seasonal_advice['icon']}")
for risk in seasonal_advice["disease_risks"]:
    st.sidebar.markdown(f"- {risk}")

st.sidebar.markdown("### Care Tips")
for tip in seasonal_advice["care_tips"]:
    st.sidebar.markdown(f"- {tip}")

# Main UI: Header
st.markdown(f"""
<div class="header">
    <h1>{t('🌿 Plant Disease Detector')}</h1>
    <p>{t('Upload a plant leaf image to detect diseases and get treatment suggestions')}</p>
</div>
""", unsafe_allow_html=True)

# Layout: Two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(f'<div class="section-title">{t("Upload Image")}</div>', unsafe_allow_html=True)
    uploaded_image = st.file_uploader(t("Choose a clear image of the plant leaf"), type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        try:
            original_image = Image.open(uploaded_image)

            st.markdown(f'<div class="section-title">{t("Image Enhancement")}</div>', unsafe_allow_html=True)
            apply_enhancement = st.checkbox(t("Apply image enhancement"), value=True)

            if apply_enhancement:
                contrast_level = st.slider(t("Contrast"), 0.5, 2.5, 1.5, 0.1)
                apply_sharpen = st.checkbox(t("Apply sharpening"), value=True)
                apply_denoise = st.checkbox(t("Reduce noise"), value=True)

                enhanced_image = enhance_leaf_image(
                    original_image, 
                    contrast=contrast_level, 
                    sharpen=apply_sharpen, 
                    denoise=apply_denoise
                )

                st.markdown(f"### {t('Original vs Enhanced Image')}")
                col_orig, col_enh = st.columns(2)
                with col_orig:
                    st.image(original_image, caption=t("Original Image"), use_column_width=True)
                with col_enh:
                    st.image(enhanced_image, caption=t("Enhanced Image"), use_column_width=True)

                image_for_analysis = enhanced_image
            else:
                st.image(original_image, caption=t("Uploaded Image"), use_column_width=True)
                image_for_analysis = original_image

            analyze_button = st.button(t("Analyze Disease"))
        except Exception as e:
            st.error(f"{t('Error opening image')}: {str(e)}")


    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="section-title">{t("Analysis Results")}</div>', unsafe_allow_html=True)

    if model and class_indices:
        if uploaded_image and 'analyze_button' in locals() and analyze_button:
            try:
                with st.spinner(t("Analyzing image...")):
                    prediction, confidence, raw_predictions = predict_image_class(model, image_for_analysis, class_indices)

                if '___' in prediction:
                    plant_type, condition = prediction.split('___')
                    display_prediction = condition.replace('_', ' ').title()
                    plant_type_display = plant_type.replace('_', ' ').title()
                else:
                    display_prediction = prediction.replace('_', ' ').title()
                    plant_type_display = t("Plant")

                is_healthy = 'healthy' in prediction.lower()
                result_class = "healthy-result" if is_healthy else "disease-result"
                result_icon = "✅" if is_healthy else "⚠️"
                result_title = t("Healthy Plant Detected") if is_healthy else f"{t(display_prediction)} {t('Detected')}"

                st.markdown(f"""
                <div class="result-box {result_class}">
                    <h3>{result_icon} {result_title}</h3>
                    <p><strong>{t('Plant type')}:</strong> {plant_type_display}</p>
                    <p><strong>{t('Confidence')}:</strong> {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

                st.progress(confidence / 100)

                if len(st.session_state.history) >= 10:
                    st.session_state.history.pop(0)

                st.session_state.history.append({
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'plant_type': plant_type_display,
                    'prediction': display_prediction,
                    'confidence': confidence,
                    'is_healthy': "Yes" if is_healthy else "No"
                })

                st.markdown(f'<div class="section-title">{t("Prediction Breakdown")}</div>', unsafe_allow_html=True)

                classes = []
                for i in range(len(raw_predictions)):
                    class_name = class_indices[str(i)]
                    if '___' in class_name:
                        classes.append(class_name.split('___')[-1].replace('_', ' ').title())
                    else:
                        classes.append(class_name.replace('_', ' ').title())

                probs = [float(p) * 100 for p in raw_predictions]
                sorted_data = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
                top_classes = [t(c) for c, _ in sorted_data[:5]]
                top_probs = [p for _, p in sorted_data[:5]]

                chart_data = pd.DataFrame({
                    t('Disease'): top_classes,
                    t('Probability (%)'): top_probs
                })

                st.bar_chart(chart_data.set_index(t('Disease')))
                
                # Treatment suggestions
                st.markdown(f'<div class="section-title">{t("Treatment Suggestions")}</div>', unsafe_allow_html=True)
                if is_healthy:
                    st.success(t("Your plant appears to be healthy! Continue with regular care:"))
                    st.markdown("""
                    - """ + t("Regular watering based on plant type") + """
                    - """ + t("Fertilize on a schedule") + """
                    - """ + t("Ensure proper light and airflow") + """
                    - """ + t("Inspect for issues weekly") + """
                    """)
                else:
                    st.warning(t("Consider these treatment options:"))
                    st.markdown("""
                    - """ + t("Isolate the plant to prevent spread") + """
                    - """ + t("Remove infected leaves") + """
                    - """ + t("Improve air circulation") + """
                    - """ + t("Avoid overhead watering") + """
                    - """ + t("Use a suitable fungicide") + """
                    """)

                    disease_key = None
                    for key in disease_info.keys():
                        if key in prediction.lower():
                            disease_key = key
                            break

                    if disease_key:
                        info = disease_info[disease_key]
                        st.markdown(f"<div class='section-title'>{t('About')} {t(display_prediction)}</div>", unsafe_allow_html=True)
                        st.markdown(f"**{t('Description')}**: {t(info['description'])}")

                        st.markdown(f"**{t('Causes')}:**")
                        for cause in info['causes']:
                            st.markdown(f"- {t(cause)}")

                        st.markdown(f"**{t('Symptoms')}:**")
                        for symptom in info['symptoms']:
                            st.markdown(f"- {t(symptom)}")

                        st.markdown(f"**{t('Prevention')}:**")
                        for prevention in info['prevention']:
                            st.markdown(f"- {t(prevention)}")

                st.markdown(f'<div class="section-title">{t("Export Report")}</div>', unsafe_allow_html=True)

                if is_healthy:
                    recommendations = [
                        t("Continue regular watering schedule"),
                        t("Maintain consistent light conditions"),
                        t("Fertilize as needed for your plant type"),
                        t("Monitor for any changes in leaf appearance")
                    ]
                else:
                    recommendations = [
                        f"{t('Treat for')} {t(display_prediction)}",
                        t("Isolate plant from others"),
                        t("Remove severely affected leaves"),
                        t("Improve air circulation"),
                        t("Follow specific treatment suggestions above")
                    ]

                    pdf = generate_pdf_report(
                        plant_type_display,
                        display_prediction,
                        confidence,
                        is_healthy,
                        recommendations,
                        normal_image=image_for_analysis,
                        enhanced_image=image_for_analysis
                    )
                    pdf_bytes = pdf.getvalue()

                    st.download_button(
                        label=t("📄 Download PDF Report"),
                        data=pdf_bytes,
                        file_name=f"plant_analysis_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                st.error(f"{t('Error during analysis')}: {str(e)}")
        else:
            st.info(t("Upload an image and click 'Analyze Disease' to start."))
    else:
        st.error(t("Model or class data failed to load."))

    st.markdown('</div>', unsafe_allow_html=True)

# Info Steps Section
st.markdown(f'<div class="section-title">{t("How to Use")}</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(f"""
    **{t('Step 1')}: {t('Upload Image')}**  
    - {t('Take a clear photo of the plant leaf')}  
    - {t('Ensure good lighting and focus')}  
    - {t('Upload using the file selector')}
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    st.markdown(f"""
    **{t('Step 2')}: {t('Analyze')}**  
    - {t('Click the "Analyze Disease" button')}  
    - {t('Wait for the detection to complete')}  
    - {t('View diagnosis and confidence')}
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col_c:
    st.markdown(f"""
    **{t('Step 3')}: {t('Take Action')}**  
    - {t('Follow the recommended treatment')}  
    - {t('Monitor plant health')}  
    - {t('Re-scan after treatment if needed')}
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# History Section
st.markdown(f'<div class="section-title">{t("Scan History")}</div>', unsafe_allow_html=True)

if st.session_state.history:
    history_data = pd.DataFrame(st.session_state.history)
    st.dataframe(history_data, use_container_width=True)

    csv = history_data.to_csv(index=False)
    st.download_button(
        label=t("Download History as CSV"),
        data=csv,
        file_name="plant_disease_history.csv",
        mime="text/csv",
    )
else:
    st.info(t("No scan history yet. Analysis results will appear here."))

# Footer
st.markdown(f"""
<div class="footer">
    <p>{t('🌿 Plant Disease Detector | For educational purposes only')}</p>
    <p>{t('Always consult a plant expert for serious infections')}</p>
</div>
""", unsafe_allow_html=True)
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model
import numpy as np
import matplotlib.pyplot as plt
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
import os
import google.generativeai as genai
from PIL import Image
import io
import html
import re

#testing_pull_request_review_agent

# Set up image parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# Set up class names
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Define model paths
XCEPTION_MODEL_PATH = 'models/xception_brain_tumor.h5'
CUSTOM_CNN_MODEL_PATH = 'models/custom_cnn_brain_tumor.h5'

def sanitize_text(text):
    """Sanitize text to prevent invalid characters in HTML/XML"""
    if not isinstance(text, str):
        text = str(text)
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    # Escape HTML special characters
    text = html.escape(text)
    # Remove any remaining control characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    return text

def safe_markdown(content):
    """Safely render markdown content"""
    try:
        sanitized_content = sanitize_text(content)
        st.markdown(sanitized_content, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error rendering content: {str(e)}")
        st.text(content)  # Fallback to plain text

def initialize_gemini():
    """Initialize Gemini with error handling"""
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-pro')
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini: {str(e)}")
        return None

def get_explanation(gemini_model, image_array, predicted_class, confidence):
    """Get concise AI explanation with 4 key observation points"""
    try:
        if gemini_model is None:
            return "Gemini model not initialized"

        # Convert numpy array to PIL Image
        image = Image.fromarray((image_array[0] * 255).astype(np.uint8))
        
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create generation config with stricter constraints
        generation_config = genai.types.GenerationConfig(
            temperature=0.3,
            top_p=0.8,
            top_k=40,
            max_output_tokens=200,
        )

        # Updated prompt to focus on descriptive observations
        prompt = f"""
        Describe the visible features in this brain MRI scan. The AI system detected patterns consistent with {sanitize_text(predicted_class)}.
        
        Provide 4 simple observations (one sentence each):
        1. Describe the prominent visible pattern or area of interest
        2. Note any distinctive intensity variations
        3. Describe the location of notable features
        4. Compare this to typical patterns seen in {sanitize_text(predicted_class)} cases
        
        Keep each point purely descriptive and observational. Format as a numbered list with exactly one line per point.
        Note: This is for research/educational purposes only.
        """

        # Create content parts
        content_parts = [prompt, image]

        try:
            response = gemini_model.generate_content(
                content_parts,
                generation_config=generation_config,
                stream=False
            )
            
            if response and response.text:
                # Clean up the response to ensure exactly 4 lines
                lines = response.text.strip().split('\n')
                formatted_lines = []
                line_count = 0
                for line in lines:
                    if line.strip() and line_count < 4:
                        formatted_lines.append(line.strip())
                        line_count += 1
                
                # If we got fewer than 4 lines, add placeholder lines
                while len(formatted_lines) < 4:
                    formatted_lines.append(f"{len(formatted_lines) + 1}. Feature observation unavailable")
                
                # Join only the first 4 lines
                return '\n'.join(formatted_lines[:4])
            else:
                return "No explanation generated"
                
        except Exception as api_error:
            return f"Error from Gemini API: {str(api_error)}"
        
    except Exception as e:
        return f"Error preparing explanation: {str(e)}"
@st.cache_resource
def load_model(model_choice):
    """Load the model with enhanced error handling"""
    try:
        model_path = XCEPTION_MODEL_PATH if model_choice == "Transfer Learning - Xception" else CUSTOM_CNN_MODEL_PATH
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model = tf_load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def generate_saliency_map(model, image):
    """Generate saliency map with error handling"""
    try:
        saliency = Saliency(model, model_modifier=lambda m: m)
        saliency_map = saliency(lambda x: x[:, tf.argmax(x[0])], image)
        return normalize(saliency_map)[0]
    except Exception as e:
        st.error(f"Error generating saliency map: {str(e)}")
        return None

def display_results(predicted_class, confidence):
    """Display classification results using Streamlit components"""
    st.markdown("""
        <style>
        .result-box {
            padding: 20px;
            background-color: #f9f9f9;
            border: 2px solid #ccc;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }
        .result-title {
            color: #ff6347;
            font-size: 24px;
            margin-bottom: 15px;
        }
        .result-content {
            color: #4CAF50;
            font-size: 20px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="result-box">
            <h2 class="result-title">Classification Result</h2>
            <h3 class="result-content">
                Predicted Class: <strong>{predicted_class}</strong>
            </h3>
            <h3 class="result-content">
                Confidence: <strong>{confidence:.2f}</strong>
            </h3>
        </div>
    """, unsafe_allow_html=True)

def display_explanation(explanation):
    """Display AI-generated explanation using Streamlit components"""
    st.markdown("""
        <style>
        .explanation-box {
            padding: 20px;
            background-color: #f0f8ff;
            border: 2px solid #4682b4;
            border-radius: 10px;
            margin-top: 20px;
        }
        .explanation-title {
            color: #4682b4;
            font-size: 22px;
            margin-bottom: 15px;
        }
        .explanation-content {
            font-size: 16px;
            line-height: 1.6;
            color: #333;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="explanation-box">
            <h2 class="explanation-title">AI-Generated Medical Explanation</h2>
            <div class="explanation-content">
                {explanation}
            </div>
        </div>
    """, unsafe_allow_html=True)

def main():
    st.title("Brain Tumor MRI Classification with AI Explanation")

    # Initialize Gemini model
    gemini_model = initialize_gemini()

    # Model selection
    model_choice = st.sidebar.radio("Choose a model:", ("Transfer Learning - Xception", "Custom CNN"))
    model = load_model(model_choice)
    
    if model is None:
        st.error("Please ensure the model files are present in the models directory")
        return

    # File upload
    uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Process image
            image_bytes = uploaded_file.read()
            image = tf.image.decode_image(image_bytes, channels=3)
            image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
            image_normalized = image / 255.0
            image_batch = np.expand_dims(image_normalized, axis=0)

            # Make prediction
            with st.spinner('Classifying image...'):
                prediction = model.predict(image_batch)
            predicted_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = np.max(prediction)

            # Display images
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_bytes, caption="Uploaded MRI Scan", width=300)
            
            with col2:
                saliency_map = generate_saliency_map(model, image_batch)
                if saliency_map is not None:
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.imshow(saliency_map, cmap='jet')
                    ax.axis('off')
                    st.pyplot(fig)
                    st.caption("Saliency Map")

            # Display classification results using the new function
            display_results(predicted_class, confidence)

            # Generate and display AI explanation
            if gemini_model:
                with st.spinner('Generating AI explanation...'):
                    explanation = get_explanation(gemini_model, image_batch, predicted_class, confidence)
                    display_explanation(explanation)

            # Display confidence metrics
            st.subheader("Confidence Level")
            st.progress(float(confidence))

            # Display class probabilities
            st.subheader("Class Probabilities")
            fig, ax = plt.subplots()
            ax.barh(CLASS_NAMES, prediction[0])
            ax.set_xlim([0, 1])
            ax.set_xlabel('Probability')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()

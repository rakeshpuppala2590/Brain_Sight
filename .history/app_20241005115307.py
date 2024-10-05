import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import requests
import os
import matplotlib.pyplot as plt
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize

# Set up Gemini API (you'll need to replace this with the actual Gemini 1.5 Flash API)
GEMINI_API_KEY = "your_gemini_api_key_here"
GEMINI_API_URL = "https://api.gemini.com/v1/explain"

# Set up paths for your dataset
TRAIN_DIR = '/Users/rakeshpuppala/Desktop/Brain_Sight/archive/Training/'
TEST_DIR = '/Users/rakeshpuppala/Desktop/Brain_Sight/archive/Testing'

# Set up image parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# Set up class names
CLASS_NAMES = ['class1', 'class2', 'class3', 'class4']  # Replace with your actual class names

def create_model():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(CLASS_NAMES), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model):
    # Your existing train_model function remains unchanged
    # ...
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )

    return model, history

def get_explanation(image, prediction):
    # This is a placeholder. You'll need to implement the actual API call to Gemini 1.5 Flash
    response = requests.post(
        GEMINI_API_URL,
        headers={"Authorization": f"Bearer {GEMINI_API_KEY}"},
        json={"image": image.tolist(), "prediction": prediction}
    )
    return response.json()["explanation"]

def generate_saliency_map(model, image):
    saliency = Saliency(model, model_modifier=lambda x: tf.keras.activations.relu(x))
    saliency_map = saliency(score_function, image)
    saliency_map = normalize(saliency_map)
    return saliency_map[0]

def score_function(output):
    return output[:, tf.argmax(output[0])]

def main():
    st.title("Brain Tumor MRI Classification")

    if 'model' not in st.session_state:
        with st.spinner("Training model..."):
            model = create_model()
            st.session_state.model, _ = train_model(model)
        st.success("Model trained successfully!")

    st.sidebar.title("Select Model")
    model_choice = st.sidebar.radio("Choose a model:", ("Transfer Learning - Xception", "Custom CNN"))

    uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = tf.image.decode_image(uploaded_file.read(), channels=3)
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
        image = image / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)

        prediction = st.session_state.model.predict(image)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction)

        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded MRI Scan", use_column_width=True)
        with col2:
            saliency_map = generate_saliency_map(st.session_state.model, image)
            plt.imshow(saliency_map, cmap='jet')
            plt.axis('off')
            st.pyplot(plt)
            st.caption("Saliency Map")

        st.subheader("Classification Result")
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")

        st.subheader("Class Probabilities")
        for class_name, prob in zip(CLASS_NAMES, prediction[0]):
            st.write(f"{class_name}: {prob:.4f}")

        if st.button("Get Explanation"):
            with st.spinner("Generating explanation..."):
                explanation = get_explanation(image[0], predicted_class)
            st.write("Explanation:", explanation)

if __name__ == "__main__":
    main()
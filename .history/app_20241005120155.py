import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize

# Set up image parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# Set up class names
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']  # Replace with your actual class names

# Set up paths for your dataset
TRAIN_DIR = '/Users/rakeshpuppala/Desktop/Brain_Sight/archive/Training/'
TEST_DIR = '/Users/rakeshpuppala/Desktop/Brain_Sight/archive/Testing'

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
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = validation_generator.samples // BATCH_SIZE

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_steps
    )

    return model, history

@st.cache_resource
def load_model():
    model = create_model()
    model, _ = train_model(model)
    return model

def generate_saliency_map(model, image):
    saliency = Saliency(model, model_modifier=lambda x: tf.keras.activations.relu(x))
    saliency_map = saliency(score_function, image)
    saliency_map = normalize(saliency_map)
    return saliency_map[0]

def score_function(output):
    return output[:, tf.argmax(output[0])]

def main():
    st.title("Brain Tumor MRI Classification")

    model = load_model()

    st.sidebar.title("Select Model")
    model_choice = st.sidebar.radio("Choose a model:", ("Transfer Learning - Xception", "Custom CNN"))

    uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = tf.image.decode_image(uploaded_file.read(), channels=3)
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
        image = image / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)

        with st.spinner('Classifying image...'):
            prediction = model.predict(image)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction)

        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded MRI Scan", use_column_width=True)
        with col2:
            with st.spinner('Generating saliency map...'):
                saliency_map = generate_saliency_map(model, image)
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

if __name__ == "__main__":
    main()
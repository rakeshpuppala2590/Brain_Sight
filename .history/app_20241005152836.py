import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
import os

# Set up image parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# Set up class names
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

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

def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0  # Normalize the image
    label = tf.one_hot(label, depth=len(CLASS_NAMES))  # One-hot encode the label
    return image, label

def create_dataset(data_dir):
    image_paths = []
    labels = []
    for class_name in CLASS_NAMES:
        class_path = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, image_name))
            labels.append(CLASS_NAMES.index(class_name))
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(image_paths)).batch(BATCH_SIZE)
    dataset = dataset.repeat()  # Add repeat to prevent running out of data
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset, len(image_paths)

def train_model(model):
    train_dataset, train_size = create_dataset(TRAIN_DIR)
    val_dataset, val_size = create_dataset(TEST_DIR)
    
    # Verify datasets
    if train_size == 0 or val_size == 0:
        st.error("Error: One or both datasets are empty.")
        return model, None
    
    steps_per_epoch = train_size // BATCH_SIZE
    validation_steps = val_size // BATCH_SIZE

    try:
        history = model.fit(
            train_dataset,
            epochs=10,
            validation_data=val_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps
        )
    except Exception as e:
        st.error(f"An error occurred during training: {str(e)}")
        return model, None

    return model, history

@st.cache_resource
def load_model():
    try:
        model = create_model()
        model, history = train_model(model)
        if history is None:
            st.warning("Model training failed. Using untrained model.")
        return model
    except Exception as e:
        st.error(f"Error in load_model: {str(e)}")
        return None

def generate_saliency_map(model, image):
    # Ensure model_modifier passes the model's output correctly
    saliency = Saliency(model, model_modifier=lambda m: m)
    saliency_map = saliency(score_function, image)
    saliency_map = normalize(saliency_map)
    return saliency_map[0]

def score_function(output):
    return output[:, tf.argmax(output[0])]

def main():
    st.title("Brain Tumor MRI Classification")

    model = load_model()
    if model is None:
        st.error("Failed to load the model. Please check the logs for details.")
        return

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
            plt.figure(figsize=(5, 5))
            plt.imshow(saliency_map, cmap='jet')
            plt.axis('off')
            st.pyplot(plt)
            st.caption("Saliency Map")

        st.subheader("Classification Result")
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")

        st.subheader("Class Probabilities")
colors = ['red', 'green', 'blue', 'orange']  # Define different colors for each class
fig, ax = plt.subplots()
ax.barh(CLASS_NAMES, prediction[0], color=colors)  # Horizontal bar graph with custom colors
ax.set_xlim([0, 1])  # Set the limit to make sure it's between 0 and 1 (probabilities)
ax.set_xlabel('Probability')
ax.set_title('Class Probabilities')
st.pyplot(fig)

if __name__ == "__main__":
    main()

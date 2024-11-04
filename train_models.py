import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# Set up constants (same as in main app)
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
TRAIN_DIR = '/Users/rakeshpuppala/Desktop/Brain_Sight/archive/Training/'
TEST_DIR = '/Users/rakeshpuppala/Desktop/Brain_Sight/archive/Testing'

def create_xception_model():
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

def create_custom_cnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0
    label = tf.one_hot(label, depth=len(CLASS_NAMES))
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
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset, len(image_paths)

def train_and_save_models():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train and save Xception model
    print("Training Xception model...")
    xception_model = create_xception_model()
    train_dataset, train_size = create_dataset(TRAIN_DIR)
    val_dataset, val_size = create_dataset(TEST_DIR)
    
    steps_per_epoch = train_size // BATCH_SIZE
    validation_steps = val_size // BATCH_SIZE
    
    history = xception_model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    
    xception_model.save('models/xception_brain_tumor.h5')
    print("Xception model saved!")
    
    # Train and save Custom CNN model
    print("\nTraining Custom CNN model...")
    custom_model = create_custom_cnn()
    history = custom_model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    
    custom_model.save('models/custom_cnn_brain_tumor.h5')
    print("Custom CNN model saved!")

if __name__ == "__main__":
    train_and_save_models()

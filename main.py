import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Set your dataset path
dataset_path = "C:\\Users\\mukes\\OneDrive\\Pictures\\Pills dataset"

# Image pre-processing
def create_data_generators(data_path, target_size=(224, 224), batch_size=32):
    return ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # 80-20 split for training and validation
    ).flow_from_directory(
        data_path,
        target_size=target_size,
        batch_size=16,
        class_mode='categorical',
        subset='training'  # Use 'validation' for validation generator
    )

train_generator = create_data_generators(dataset_path)
num_classes = len(train_generator.class_indices)

# Build model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    epochs=100,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)]
)

# Save the model
model.save('pill_identifier_model.h5')

tf.keras.backend.clear_session()


# Import necessary libraries
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import json

# Define paths
train_dir = 'C:/Users/user/OneDrive/Desktop/Deep_Learning_Project/Fruit Disease Detection Dataset.v4-presentation.tensorflow/train'
valid_dir = 'C:/Users/user/OneDrive/Desktop/Deep_Learning_Project/Fruit Disease Detection Dataset.v4-presentation.tensorflow/valid'
test_dir = 'C:/Users/user/OneDrive/Desktop/Deep_Learning_Project/Fruit Disease Detection Dataset.v4-presentation.tensorflow/test'

# Image dimensions and batch size
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescale validation and test data
valid_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load datasets
train_data = train_datagen.flow_from_directory(
    train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical'
)

valid_data = valid_datagen.flow_from_directory(
    valid_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical'
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=20,
    steps_per_epoch=len(train_data),
    validation_steps=len(valid_data)
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model
model_save_path = 'C:/Users/user/OneDrive/Desktop/Deep_Learning_Project/Fruit Disease Detection Dataset.v4-presentation.tensorflow/model/fruit_disease_detector.h5'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print("Model saved at:", model_save_path)

# Save the class indices mapping
class_indices_path = 'C:/Users/user/OneDrive/Desktop/Deep_Learning_Project/Fruit Disease Detection Dataset.v4-presentation.tensorflow/class_indices.json'
with open(class_indices_path, 'w') as f:
    json.dump(train_data.class_indices, f)
print("Class indices saved at:", class_indices_path)

# Plot accuracy and loss
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.show()

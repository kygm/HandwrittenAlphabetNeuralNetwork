import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import logging
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from PIL import Image

parsed_images_dir = os.path.join(os.getcwd(), "parsedImages")
os.makedirs(parsed_images_dir, exist_ok=True)

# Suppress TensorFlow warnings
tf.get_logger().setLevel(logging.ERROR)
tf.random.set_seed(7)

# Constants
EPOCHS = 8
BATCH_SIZE = 128
IMG_HEIGHT, IMG_WIDTH = 28, 28
NUM_CLASSES = 26  # A–Z
NUMBER_OF_NEURONS = 200  # Number of neurons in the hidden layer

# Character lookup: EMNIST "letters" labels go from 1 to 26 for A-Z
label_map = {i: chr(ord('A') + i - 1) for i in range(1, 27)}

#This gets the dataset to train the neural network on
print("Loading EMNIST Letters dataset...")
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

def convert_to_numpy(dataset):
    images = []
    labels = []
    for img, label in tfds.as_numpy(dataset):
        img = np.rot90(img, k=3)
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

train_images, train_labels = convert_to_numpy(ds_train)
test_images, test_labels = convert_to_numpy(ds_test)

#Normalizing the images
mean = np.mean(train_images)
std = np.std(train_images)
train_images = (train_images - mean) / std
test_images = (test_images - mean) / std

train_labels -= 1
test_labels -= 1
train_labels = to_categorical(train_labels, NUM_CLASSES)
test_labels = to_categorical(test_labels, NUM_CLASSES)

val_split = int(len(test_images) * 0.8)
val_images = test_images[:val_split]
val_labels = test_labels[:val_split]
test_images = test_images[val_split:]
test_labels = test_labels[val_split:]

#Training the neural network
initializer = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(NUMBER_OF_NEURONS, activation='relu', kernel_initializer=initializer),
    keras.layers.Dense(NUM_CLASSES, activation='softmax', kernel_initializer=initializer)
])

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\nTraining model on EMNIST Letters...")
history = model.fit(
    train_images, train_labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(val_images, val_labels),
    verbose=2,
    shuffle=True
)

def predict_custom_images_with_gui():

    folder_path = os.path.join(os.getcwd(), "MyHandwrittenLetters")

    print(f"\nScanning folder: {folder_path}")
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    if not files:
        print("No image files found in the folder.")
        return

    predictions_path = os.path.join(os.getcwd(), "predictions.txt")
    print(f"\nPredictions will be written to {predictions_path}")

    correct = 0
    total = 0

    with open(predictions_path, "w") as output_file:
        for filename in files:
            filepath = os.path.join(folder_path, filename)
            try:
                # Extract true letter from filename - expected format is x_1.png etc
                true_letter = next((c.lower() for c in filename if c.isalpha()), None)
                if not true_letter:
                    raise ValueError("Couldn't extract letter from filename")

                # Load and preprocess
                img = image.load_img(filepath, color_mode='grayscale')
                img = img.resize((IMG_WIDTH, IMG_HEIGHT))
                img_array = image.img_to_array(img).squeeze()

                img_array = 255 - img_array      # Invert color
                img_array = np.fliplr(img_array) # Mirror

                # Saving out the parsed image
                output_img_path = os.path.join(parsed_images_dir, f"parsed_{filename}")
                Image.fromarray(img_array.astype(np.uint8)).save(output_img_path)

                # Normalize and predict
                img_array = (img_array - mean) / std
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array, verbose=0)
                predicted_class = np.argmax(prediction)
                predicted_char = chr(ord('a') + predicted_class)

                result = f"\"{filename}\" - predicted '{predicted_char}'"
                if predicted_char == true_letter:
                    correct += 1
                total += 1

                print(result)
                output_file.write(result + "\n")

                # Showing the parsed image for debugging
                parsed_img_debug = img_array.squeeze() * std + mean  # unnormalize for display
                plt.figure(figsize=(2.5, 2.5))
                plt.imshow(parsed_img_debug, cmap='gray')
                plt.title(f"Parsed {filename}")
                plt.axis('off')
                plt.tight_layout()
                plt.show()

            except Exception as e:
                error_msg = f"Could not process {filename}: {e}"
                print(error_msg)
                output_file.write(error_msg + "\n")

        # Final accuracy report
        if total > 0:
            accuracy = (correct / total) * 100
            summary = f"\nAccuracy: {correct}/{total} = {accuracy:.2f}%"
            print(summary)
            output_file.write(summary + "\n")


def evaluate_emnist_test_set():
    print("\nEvaluating on 10 samples from the EMNIST test set...\n")
    for i in range(10):
        img = test_images[i]
        label_index = np.argmax(test_labels[i])
        true_char = chr(ord('a') + label_index)

        # Model expects a batch dimension
        img_input = img.reshape(1, 28, 28)
        prediction = model.predict(img_input, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_char = chr(ord('a') + predicted_index)

        print(f"Test letter actual value: '{true_char}' - predicted '{predicted_char}'")

        # --- Plot image with true vs predicted label ---
        plt.figure(figsize=(2.5, 2.5))
        plt.imshow(img, cmap='gray')
        plt.title(f"True: '{true_char}' | Pred: '{predicted_char}'")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

print("\nTraining complete. Choose how to test the model:")
print("1. Predict using your own grayscale images (any size, auto-converted to 28x28)")
print("2. Evaluate using held-out EMNIST letters test set (first 10 samples)")
choice = input("Enter 1 or 2: ").strip()

again = True
while again:
    if choice == '1':
        predict_custom_images_with_gui()
    elif choice == '2':
        evaluate_emnist_test_set()
    else:
        print("Invalid choice. Please enter 1 or 2.")

    again = input("\nDo you want to test again? (y/n): ").strip().lower() == 'y'

    if again:
        choice = input("Enter 1 or 2: ").strip()

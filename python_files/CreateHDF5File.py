import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Function to calculate the area of white objects in an image
def calculate_area_of_white_object(image_gray):
    _, binary_image = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(contour) for contour in contours)
    return total_area

# Function to load and preprocess images from a folder
def load_images_from_folder(folder_path, target_size=(100, 100)):
    images = []
    areas = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        image = cv2.imread(path)
        if image is not None:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_image, target_size)  # Resize the image
            images.append(resized_image)
            area = calculate_area_of_white_object(gray_image)
            areas.append(area)
            print(filename,"-",area)
    return np.array(images), np.array(areas)

# Define folder path containing images
folder_path = r"D:\DATASET2"  # Provide the path to your images folder

# Load images and calculate areas
images, areas = load_images_from_folder(folder_path)

# Define labels based on area thresholds
classifications = [
    (700000, 850000, "W180"),
    (600000, 690000, "W210"),
    (500000, 600000, "W320"),
    (250000, 400000, "W400")
]
labels = []
for area in areas:
    for threshold_area in classifications:
        if threshold_area[0] <= area < threshold_area[1]:
            labels.append(threshold_area[2])
            break
    else:
        labels.append(classifications[-1][2])

# Convert class labels to numerical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# Convert numerical labels to one-hot encoded format
num_classes = len(np.unique(y_encoded))
y_one_hot = np.eye(num_classes)[y_encoded]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, y_one_hot, test_size=0.2, random_state=42)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Preprocess data (reshape and normalize)
X_train = X_train.reshape((-1, 100, 100, 1)) / 255.0
X_test = X_test.reshape((-1, 100, 100, 1)) / 255.0

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save(r"D:\RobinHDFile.h5")

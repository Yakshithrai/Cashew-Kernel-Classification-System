import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import time
import requests
import serial

se=serial.Serial('COM9',9600)
#time.sleep(2)


def send_angle(angle):
    se.write(angle.encode())
    print("Sent angle:", angle)



# Function to calculate the area of white objects in an image
def calculate_area_of_white_object(image_gray):
    _, binary_image = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(contour) for contour in contours)
    return total_area

# Function to classify cashew based on the model prediction
def classify_cashews_in_folder(folder_path, model):
    # Iterate through all images in the folder
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        
        # Load and preprocess the input image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, (100, 100))
        normalized_image = resized_image.reshape((1, 100, 100, 1)) / 255.0

        # Predict the class probabilities
        probabilities = model.predict(normalized_image)
        
        # Get the predicted class label
        predicted_class_index = np.argmax(probabilities)
        predicted_class = classifications[predicted_class_index]

        # Get the area of white objects in the image
        area = calculate_area_of_white_object(image)
        
        # Print the prediction and area for each image
        print("Image:", filename)
        print("Prediction:", predicted_class)
        #time.sleep(5)
        if predicted_class=="W400":
            send_angle('D')
        else:
            send_angle('C')
        #time.sleep(2)
        #send_angle('A')
            
            
            
            
        #print("Area of white object:", area)
        print()

# Path to the folder containing test images

test_folder_path = "D:\TEST2\Test"

    # Load the trained model
model = load_model(r"D:\newRobinHDFile.h5")

    # Define classifications
classifications = ["W180", "W210", "W320", "W400"]
def test_code():

    # Classify cashews in the folder
    classify_cashews_in_folder(test_folder_path, model)




print('Loading...')
send_angle('A')

def capture():
    url='http://192.168.136.17/capture?_cb=1714372541111'
    filename = 'D:\TEST2\Test\catured_image.jpg'
    response=requests.get(url)
    if response.status_code==200:
        with open(filename,'wb') as f:
            f.write(response.content)
        print(f"Image saved as {filename}")
        
    else:
        print(f"Failed to download Image")


while True:
    if se.in_waiting > 0:
        ard_input = se.readline().decode().strip()
        if ard_input == '0':
            print(ard_input)
            time.sleep(2)
            capture()
            test_code()
           

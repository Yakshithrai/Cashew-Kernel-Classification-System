import os
import cv2

def calculate_area_of_white_object(image_gray):
    # Threshold the image to get binary image
    ret, binary_image = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate total area of white objects
    total_area = 0
    for contour in contours:
        total_area += cv2.contourArea(contour)

    return total_area

def load_images_from_folder(folder_path):
    image_areas = []
    
    for filename in os.listdir(folder_path):
        # Check if the file is an image (you can add more extensions if needed)
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # Construct the full path to the image file
            img_path = os.path.join(folder_path, filename)
            # Load the image using OpenCV
            img = cv2.imread(img_path)
            if img is not None:
                # Convert image to grayscale
                image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                area = calculate_area_of_white_object(image_gray)
                image_areas.append((filename, area))
                print(filename,"-",area)
                
    return image_areas

# Example usage:
folder_path = r"C:\Users\ROBIN\Desktop\Cashew"
image_areas = load_images_from_folder(folder_path)

# Print area of each image
max_area = float('-inf')
min_area = float('inf')
max_area_filename = ""
min_area_filename = ""
total_area = 0

for filename, area in image_areas:
    #print(f"Area of {filename}: {area}")
    total_area += area
    if area > max_area:
        max_area = area
        max_area_filename = filename
    if area < min_area:
        min_area = area
        min_area_filename = filename

# Calculate average area
num_images = len(image_areas)
average_area = total_area / num_images

print(f"Maximum area: {max_area} (from {max_area_filename})")
print(f"Minimum area: {min_area} (from {min_area_filename})")
#print(f"Average area: {average_area}")

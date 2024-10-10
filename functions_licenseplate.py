import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import random

def display_and_save_cropped_plates(cropped_plates_dict, save_folder):
    """
    Displays the cropped license plate images using matplotlib (plt) and saves them to the specified folder.

    Parameters:
    - cropped_plates_dict (dict): Dictionary where keys are image filenames and values are lists of cropped license plate images.
    - save_folder (str): Folder path where the cropped images will be saved.
    """
    # Ensure the save directory exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for image_file, cropped_plates in cropped_plates_dict.items():
        for i, cropped_plate in enumerate(cropped_plates):
            if i == 0:  # avoid getting other plates from the image, just the one we are interested in
                plt.figure(figsize=(5, 3))
                plt.imshow(cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB))  
                plt.title(f'Cropped Plate from {image_file}')
                plt.axis('off')  # Hide axis for better visualization
                plt.show()

                # Save the cropped image to the specified folder
                save_path = os.path.join(save_folder, f"{image_file}")
                cv2.imwrite(save_path, cropped_plate)
                print(f"Saved cropped plate to {save_path}")
        
#to show the images
def show_image(image, title="Image"):
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)
    plt.show()

#plate recognition
def plate_recognition(image_folder):
    file_list = os.listdir(image_folder)
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [file for file in file_list if os.path.splitext(file)[1].lower() in image_extensions]
    random_image = random.choice(image_files)
    #full path
    random_image_path = os.path.join(image_folder, random_image)
    image = cv2.imread(random_image_path)

    #define range of blue europe license plate
    lower_blue = np.array([100, 100, 100])  # Limit blue inferior
    upper_blue = np.array([140, 255, 255])  # Limit blue superior 

    #HSV space color
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #create a mask for blues
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    #find the countors
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Process each countour found
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # minimum range
            x, y, w, h = cv2.boundingRect(contour)

            #height must be at least 1.4 times greater than the width
            if h >= 1.4 * w:
                #to capture the entire license plate
                extended_width = 520  #Total width of the license plate in mm
                scale_factor = extended_width / 40  #we assume the width of the detected blue rectangle is 40 mm
                new_width = int(w * scale_factor)

                new_x = x
                new_y = y
                if new_x + new_width >= image.shape[1]:
                    new_width = image.shape[1] - new_x  #adjust the width to not exceed the image

                #region of interest
                roi = image[new_y:new_y + h, new_x:new_x + new_width]

                #show
                show_image(roi, title=f"License Plate Region: {random_image_path}")
                print(f"File: {random_image_path}, Coordinates: ({new_x}, {new_y}, {new_width}, {h}), Region dimensions: {roi.shape}")
    

    


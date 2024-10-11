import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import random
from ultralytics import YOLO

"""

FIRST LICENSE PLATE (EL DE LA MERI)

"""

# Function to visualize images side by side
def visualize(images, titles, suptitle=None, cmap=None):
    plt.figure(figsize=(15, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i], cmap=cmap if i == 1 else 'gray')  # Show corners in color, grayscale in original
        plt.title(titles[i])
        plt.axis('off')
    if suptitle:
        plt.suptitle(suptitle)
    plt.show()

def detect_and_crop_license_plate(binary_image, original_image):
    """
    Detects and crops the license plate from the given images.

    Parameters:
    - binary_image: Preprocessed binary image.
    - original_image: The original input image.

    Returns:
    - license_plate: Cropped image of the detected license plate or None if not found.
    - img_with_contour: Image of the car with the detected license plate contour drawn or None if not found.
    """
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on area in descending order and take top 10
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    license_plate_contour = None

    for contour in contours:
        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

        # Check if the approximated contour has 4 points (quadrilateral)
        if len(approx) == 4:
            # Compute the bounding box and aspect ratio
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Define a range for aspect ratio typical of license plates
            if 2 <= aspect_ratio <= 6:
                # Further check for parallelism of edges
                angles = []
                for i in range(4):
                    pt1 = approx[i][0]
                    pt2 = approx[(i + 1) % 4][0]
                    pt3 = approx[(i + 2) % 4][0]

                    v1 = pt2 - pt1
                    v2 = pt3 - pt2

                    angle = angle_between(v1, v2)
                    angles.append(angle)

                if all(80 <= angle <= 100 for angle in angles):
                    license_plate_contour = approx
                    break  # Stop once the likely license plate is found

    if license_plate_contour is not None:
        # Draw the detected contour on the image (for visualization)
        img_copy = original_image.copy()
        cv2.drawContours(img_copy, [license_plate_contour], -1, (0, 255, 0), 3)

        # Obtain a top-down view of the license plate using perspective transform
        license_plate = four_point_transform(original_image, license_plate_contour.reshape(4, 2))

        # Display the results
        # Use cv2_imshow if in Jupyter; otherwise, use cv2.imshow
        try:
            cv2.imshow(license_plate)
            cv2.imshow(img_copy)
        except ImportError:
            cv2.imshow("Cropped License Plate", license_plate)
            cv2.imshow("Detected License Plate", img_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print("License plate detected")
        # Return both the cropped license plate and the image with contour
        return license_plate, img_copy

    else:
        print("License plate not found.")
        return None, None

def angle_between(v1, v2):
    """
    Calculates the angle between two vectors in degrees.
    """
    v1 = v1.astype(float)
    v2 = v2.astype(float)
    unit_v1 = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) != 0 else v1
    unit_v2 = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) != 0 else v2
    dot_product = np.dot(unit_v1, unit_v2)
    # Clamp the dot_product to avoid numerical issues with arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle = np.arccos(dot_product)
    return np.degrees(angle)

def order_points(pts):
    """
    Orders points in the order: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum and difference to find corners
    s = np.sum(pts, axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left

    return rect

def four_point_transform(image, pts):
    """
    Applies a perspective transform to obtain a top-down view of the detected license plate.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

"""

SECOND LICENSE PLATE (EL DE LA LARA)

"""

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
    
"""

YOLO LICENSE PLATE (EL DE LA LARA)

"""

def boundingbox(folder_path):
    # Load models
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('YOLO_Files/license_plate_detector.pt')
    """
    Detects license plates in all images within a folder and returns the cropped license plate images.

    Parameters:
    - folder_path (str): The path to the folder containing image files.

    Returns:
    - results (dict): A dictionary where keys are image filenames and values are lists of cropped license plate images.
    """
    # Dictionary to hold results: {image_filename: [list_of_cropped_license_plate_images]}
    results = {}

    # Loop through all images in the folder
    for image_file in os.listdir(folder_path):
        if image_file.endswith(('.jpg')):  # Ensure we're processing image files only
            image_path = os.path.join(folder_path, image_file)
            frame = cv2.imread(image_path)

            if frame is None:
                print(f"Error loading image: {image_file}")
                continue  # Skip images that can't be loaded

            # Suppress unnecessary logging by the YOLO model
            license_plates = license_plate_detector(frame, verbose=False)[0]  # Set verbose to False

            # List to store cropped license plates for the current image
            cropped_license_plates = []

            # Crop and save all detected license plates
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                cropped_plate = frame[int(y1):int(y2), int(x1):int(x2)]  # Crop the detected license plate
                cropped_license_plates.append(cropped_plate)  # Add to list

            # Only save results if any license plates were detected
            if cropped_license_plates:
                results[image_file] = cropped_license_plates

    return results    





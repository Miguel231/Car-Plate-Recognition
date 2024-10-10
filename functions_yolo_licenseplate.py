from ultralytics import YOLO
import cv2
import os

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
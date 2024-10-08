from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
import random
from YOLO_Files.util import get_car, read_license_plate

def yolo_plate_recognition(image_folder):
    # Load models
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('YOLO_Files\license_plate_detector.pt')

    # Specify the vehicle classes you want to detect
    vehicles = [2, 3, 5, 7]  # 2: Automóvil;3: Motocicleta;5: Autobús;7: Camión

    file_list = os.listdir(image_folder)
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [file for file in file_list if os.path.splitext(file)[1].lower() in image_extensions]
    random_image = random.choice(image_files)
    #full path
    random_image_path = os.path.join(image_folder, random_image)
    frame = cv2.imread(random_image_path)

    # Detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []
    blue_rectangle = None  # Variable to hold the blue rectangle coordinates

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection


        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

            # Draw bounding box around vehicles
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Detect license plates
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Crop and process license plate
        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

        # Draw bounding box around license plates
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # Assuming the blue rectangle corresponds to a license plate or vehicle detection
    if len(license_plates.boxes.data.tolist()) > 0:
        # Get the first detected license plate as blue rectangle
        blue_rectangle = license_plates.boxes.data.tolist()[0]  # Change index if needed
        x1, y1, x2, y2, score, class_id = blue_rectangle

        # Crop the blue rectangle region
        blue_region = frame[int(y1):int(y2), int(x1):int(x2)]

        # Display only the cropped blue rectangle using Matplotlib
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(blue_region, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Hide axes
        plt.title(f'Blue Rectangle - {random_image}')  # Display image title
        plt.show()



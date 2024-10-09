import functions_licenseplate as fl
import functions_yolo_licenseplate as yolo


image_folder = 'G:/Mi unidad/LICENSE_PLATES_RECOGITION_L&V/combined_folder_last(onlyspain)'

#fl.plate_recognition(image_folder= image_folder)
yolo.yolo_plate_recognition(image_folder=image_folder)
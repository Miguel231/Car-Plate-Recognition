import functions_licenseplate as fl
import functions_yolo_licenseplate as yolo


image_folder = 'G://.shortcut-targets-by-id//1xjrivG-T7lph1wnu1KGxnsESEs0U5vvV//LICENSE_PLATES_RECOGITION_L&V//combined_folder_last(onlyspain)'

#fl.plate_recognition(image_folder= image_folder)
yolo.yolo_plate_recognition(image_folder=image_folder)
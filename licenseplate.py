import functions_licenseplate as fl

from google.colab import drive
drive.mount('/content/drive')

image_folder = '/content/drive/MyDrive/LICENSE_PLATES_RECOGITION_L&V/combined_folder_last(onlyspain)'

fl.plate_recognition(image_folder= image_folder)
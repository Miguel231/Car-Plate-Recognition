import cv2
import functions_licenseplate as f
import os
import pandas as pd
import easyocr
import matplotlib.pyplot as plt

def count_files_in_folders(starting_path, output_excel):
    folder_data = []

    for folder_name in os.listdir(starting_path):
        try:
            current_path = os.path.join(starting_path, folder_name)

            if os.path.isdir(current_path):
                files_in_folder = [f for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f))]

                if len(files_in_folder) > 0:
                    folder_data.append({
                        'Folder': current_path[-1],
                        'Number of Files': len(files_in_folder)
                    })

        except Exception as e:
            print(f"Error processing {folder_name}: {e}")

    df = pd.DataFrame(folder_data)
    df.to_csv(output_excel, index=False)

    print(f"Data has been saved to {output_excel}")

def OCR_image(license_plate, t = 120, min_h = 80, min_w = 15, min_ar = 0.8, max_ar = 1.2, area = 5000):
    if license_plate is None or license_plate.size == 0:
        print("Error: The license plate image is empty or not loaded correctly.")
        return None

    upscale_factor = 8
    upscaled_license_plate = cv2.resize(license_plate, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)
    

    gray = cv2.cvtColor(upscaled_license_plate, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                #  cv2.THRESH_BINARY_INV, 11, 2)

    #dilated operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    f.visualize([dilated],
                ["Dilated"], cmap='gray')
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])

    characters = []
    for contour in sorted_contours:
        hull = cv2.convexHull(contour)
        x, y, w, h = cv2.boundingRect(hull)
        #x1, y1, w1, h1 = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        area = cv2.contourArea(contour)

        if (h > min_h and w > min_w) and (aspect_ratio >= min_ar and aspect_ratio <= max_ar) and area > 5000:
            char = dilated[y:y+h, x:x+w]
            characters.append(char)

            cv2.rectangle(upscaled_license_plate, (x, y), (x + w, y + h), (0, 255, 0), 2)
        

    f.visualize([upscaled_license_plate],
                ["Segmented Characters"], cmap='gray')
    
    return upscaled_license_plate, characters

"""

YOUTUBE LICENSE PLATE (EL DEL MIGUEL)

"""
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Find contours and filter them based on size constraints
def find_contours(dimensions, img):
    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on the provided size constraints
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    # Sort contours based on contour area, taking the largest 15
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread('PngItem_1231666.png')

    x_cntr_list = []
    img_res = []
    for cntr in cntrs:
        intx, inty, intwidth, intHeight = cv2.boundingRect(cntr)
        if lower_width < intwidth < upper_width and lower_height < intHeight < upper_height:
            x_cntr_list.append(intx)

            char_copy = np.zeros((44,24))
            char = img[inty:inty+intHeight, intx:intx+intwidth]
            char = cv2.resize(char, (20, 40))

            cv2.rectangle(ii, (intx,inty),(intwidth+intx, inty+intHeight),(50,21,200),2)
            plt.imshow(ii,cmap="gray")
            
            char = cv2.subtract(255, char)

            char_copy[2:42,2:22] = char
            char_copy[0:2, : ] = 0
            char_copy[:, 0:2 ] = 0
            char_copy[42:44, : ] = 0
            char_copy[:, 22:24 ] = 0
            
            # Store character image
            img_res.append(char_copy)
    
    # Return sorted contours based on the x-coordinate (left-to-right)
    indices = sorted(range(len(x_cntr_list)),key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])
    img_res = np.array(img_res_copy)
    
    return img_res

# Segment characters from the license plate image
def segment_characters(image):
    # Resize image to fixed dimensions
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    # Apply Otsu's threshold to get a binary image
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Erode and dilate image to remove noise
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))
    
    # Get dimensions of each character (tuned to typical license plate character sizes)
    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255
    
    dimensions = [LP_WIDTH/6, LP_WIDTH/2, LP_HEIGHT/10, 2*LP_HEIGHT/3]
    
    plt.axis("off")
    plt.title("Binary")
    plt.imshow(img_binary_lp, cmap='gray')
    plt.show()

    # Find contours corresponding to characters
    char_list = find_contours(dimensions, img_binary_lp)
    
    return char_list

"""

EASY OCR

"""

import easyocr
import cv2
from PIL import Image
from PIL import ImageOps

def easy_ocr_method(license_plate_image):
    # Initialize EasyOCR Reader
    reader = easyocr.Reader(['en'])

    # Use EasyOCR to detect text
    results = reader.readtext(license_plate_image)

    # Print the detected results
    license_plate_text = ""
    for (bbox, text, prob) in results:
        print(f"Detected Text: {text}, Probability: {prob}")
        license_plate_text += text

    return license_plate_text

def best_segmentation_method(method1, method2):

    # Almacenar los resultados con la cantidad de caracteres detectados
    results = [
        (method1, len(method1)),
        (method2, len(method2))
        #,(result3, len(result3))
    ]

    valid_results = [result for result in results if result[1] >= 7]

    if valid_results:
        return min(valid_results, key=lambda x: x[1])[0]

    return max(results, key=lambda x: x[1])[0]
    
def filter_spain_plates(spain):
    number_to_letter = {'6': 'G', '8': 'B', 
                        '1': 'I', '2': 'Z', '0': 'O'}
    
    letter_to_number = {'G': '6', 'B': '8', 
                        'I': '1', 'Z': '2', 'O': '0'}
    
    if "AEIOU" in spain[-3:]:
        correct = "False Spanish License Plate, don't use API"
        print(correct)
        return correct
    
    else:
        #separate the string
        if len(spain) == 8:
            numbers = spain[1:4]
            letters = spain[4:]

            correct = "E - "
        elif len(spain) == 7:
            #separate the string
            numbers = spain[:4]
            letters = spain[4:]

            correct = "E - "
        else:
            numbers = spain[-7:-3]
            letters = spain[-3:]

            correct = "E - "

        #filter
        correct_letters = ''.join([number_to_letter.get(char, char) for char in letters])
        correct_numbers = ''.join([letter_to_number.get(char, char) for char in numbers])
        
        #combine
        correct = correct + correct_numbers + correct_letters

        return correct



"""
path = 'G://.shortcut-targets-by-id//1xjrivG-T7lph1wnu1KGxnsESEs0U5vvV//LICENSE_PLATES_RECOGITION_L&V//GITHUB_trainset_croppedimages//6011HHV.jpg'
image = cv2.imread(path)
car = image.copy()
characters = OCR_image(license_plate = image, t = 200 ,min_h = 80, min_w = 15, min_ar = 0.2, max_ar = 1)

"""
import os
import matplotlib.pyplot as plt
import numpy as np


def load_ground_truth_from_filenames(image_dir):
    ground_truth = []
    for image_file in os.listdir(image_dir):
        if image_file.endswith('.jpg'):
            label = os.path.splitext(image_file)[0]  # Get the label by removing the file extension
            ground_truth.append(label)
    
    # Sort the list to maintain a consistent order
    ground_truth.sort()
    return ground_truth


# Function to load predictions from a .txt file
def load_predictions_from_txt(txt_file):
    predictions = []
    with open(txt_file, 'r') as file:
        for line in file:
            prediction = line.strip()  # Remove whitespace around the line
            # Filter out the unwanted characters
            if prediction.startswith("E -") or prediction.startswith("E-") or prediction.startswith("E - ") or prediction.startswith("E- ") or prediction.startswith("€") :
                prediction = prediction.replace("E -", "").replace("E-", "").replace("E - ", "").replace("E- ", "").strip()
                
            if len(prediction) > 7:
                prediction = prediction[:7].upper()  
            elif len(prediction) == 7:
                prediction = prediction.upper() 

            predictions.append(prediction)

    return predictions 


def evaluate_predictions(ground_truth, predictions, f, num):
    c = 0
    fo = 0
    for gt, pred in zip(ground_truth, predictions):
        if num == 0:
            if gt == pred:
                f.append(1)
                fo+=1
            else:
                f.append(0)   
        else:
            if gt == pred:
                if f[c] == 0:
                    f[c] == 1
                    fo+=1
        c+=1


    print("TRUE:",fo)
    print("TOTAL:",c)
    num = 1
    return fo,num

def evaluate_predictions(ground_truth, predictions, f, num):
    c = 0
    fo = 0
    n=0
    for gt, pred in zip(ground_truth, predictions):
        if num == 0:
            if gt == pred:
                f.append(1)
                fo+=1
                n+=1
            else:
                f.append(0)   
        else:
            if gt == pred:
                n+=1
                if f[c] == 0:
                    f[c] = 1
                    fo+=1
        c+=1


    print("DETECTED:", n)
    print("NOT DETECTED PREVIOUSLY::",fo)
    num = 1
    return fo,num


def evaluate_predictions_characters(ground_truth, predictions, f, num):
    c = 0
    fo = 0
    n = 0
    for gt, pred in zip(ground_truth, predictions):
        if num == 0:
            if len(pred) < 7:
                for char_pred in pred:
                    if char_pred in gt:
                        f.append(1)
                        fo += 1
                        n+=1
                    else:
                        f.append(0)
                    c += 1
                
                # Calculate how many zeros are needed to pad the list f to length 7
                zeros_to_add = 7 - len(pred)
                f.extend([0] * zeros_to_add)
                c += zeros_to_add  
            else:
                for char_pred in pred:
                    if char_pred in gt:
                        f.append(1)
                        fo+=1
                        n+=1
                    else:
                        f.append(0)
                    c+=1   
        else:
            for char_pred in pred:
                if char_pred in gt:
                    n+=1
                    if f[c] == 0:
                        f[c] = 1
                        fo+=1
                c+=1



    print("DETECTED:", n)
    print("NOT DETECTED PREVIOUSLY::",fo)
    num = 1
    return fo,num


def run_evaluation_with_filenames(image_dir, svm_txt, cnn_txt, ocr_txt, svm_txt_fil, cnn_txt_fil, ocr_txt_fil,f):
    ground_truth = load_ground_truth_from_filenames(image_dir)
    svm_predictions = load_predictions_from_txt(svm_txt)
    cnn_predictions = load_predictions_from_txt(cnn_txt)
    ocr_predictions = load_predictions_from_txt(ocr_txt)
    svm_predictions_fil = load_predictions_from_txt(svm_txt_fil)
    cnn_predictions_fil = load_predictions_from_txt(cnn_txt_fil)
    ocr_predictions_fil = load_predictions_from_txt(ocr_txt_fil)
    suma = 0
    num = 0
    print("LICENSE PLATES DETECTED: ")
    print("\n")
    print("SVM:")
    fo,num = evaluate_predictions(ground_truth, svm_predictions, f, num)
    suma+=fo
    print("\n")
    print("CNN:")
    fo,num=evaluate_predictions(ground_truth, cnn_predictions, f, num)
    suma+=fo
    print("\n")
    print("OCR:")    
    fo,num=evaluate_predictions(ground_truth, ocr_predictions, f, num)
    suma+=fo
    print("\n")
    print("SVM_FILTER:")    
    fo,num=evaluate_predictions(ground_truth, svm_predictions_fil, f, num)
    suma+=fo
    print("\n")
    print("CNN_FILTER:")
    fo,num=evaluate_predictions(ground_truth, cnn_predictions_fil, f, num)
    suma+=fo
    print("\n")
    print("OCR_FILTER:")    
    fo, num = evaluate_predictions(ground_truth, ocr_predictions_fil, f, num)
    suma+=fo
    print("\n")
    print("TOTAL LICENSE PLATES DETECTED CORRECTLY: ", suma)   
    print("\n")
    print("\n") 


    ### BY CHARACTERS ##################################
    suma = 0
    num = 0
    print("CHARACTERS DETECTED: ")
    print("\n")
    print("SVM:")
    fo,num = evaluate_predictions_characters(ground_truth, svm_predictions, f, num)
    suma+=fo
    print("\n")
    print("CNN:")
    fo,num=evaluate_predictions_characters(ground_truth, cnn_predictions, f, num)
    suma+=fo
    print("\n")
    print("OCR:")    
    fo,num=evaluate_predictions_characters(ground_truth, ocr_predictions, f, num)
    suma+=fo
    print("\n")
    print("SVM_FILTER:")    
    fo,num=evaluate_predictions_characters(ground_truth, svm_predictions_fil, f, num)
    suma+=fo
    print("\n")
    print("CNN_FILTER:")
    fo,num=evaluate_predictions_characters(ground_truth, cnn_predictions_fil, f, num)
    suma+=fo
    print("\n")
    print("OCR_FILTER:")    
    fo, num = evaluate_predictions_characters(ground_truth, ocr_predictions_fil, f, num)
    suma+=fo
    print("\n")
    print("TOTAL CHARACTERS DETECTED CORRECTLY: ", suma)   


import matplotlib.pyplot as plt
import numpy as np

def plot_detection_accuracies(plates_detected, plates_not_detected_previously, total_license_plates,
                              characters_detected, characters_not_detected_previously, total_characters,
                              total_license_plates_correct, total_characters_correct):
    methods = ['SVM', 'CNN', 'OCR', 'SVM_FILTER', 'CNN_FILTER', 'OCR_FILTER']


    license_plate_accuracy = [(detected / total_license_plates) * 100 for detected in plates_detected]
    character_accuracy = [(detected / total_characters) * 100 for detected in characters_detected]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(methods))  # Label locations


    axs[0].bar(x - 0.2, plates_detected, width=0.4, label='Detected')
    axs[0].bar(x + 0.2, plates_not_detected_previously, width=0.4, label='Not Detected Previously')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(methods)
    axs[0].set_ylabel('Number of License Plates')
    axs[0].set_title('License Plates Detected')
    axs[0].legend()


    axs[0].text(0.5, max(plates_detected) + 5, f'Total Correct: {total_license_plates_correct}', 
                fontsize=12, ha='center')

    axs[1].bar(x - 0.2, characters_detected, width=0.4, label='Detected')
    axs[1].bar(x + 0.2, characters_not_detected_previously, width=0.4, label='Not Detected Previously')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(methods)
    axs[1].set_ylabel('Number of Characters')
    axs[1].set_title('Characters Detected')
    axs[1].legend()


    axs[1].text(0.5, max(characters_detected) + 50, f'Total Correct: {total_characters_correct}', 
                fontsize=12, ha='center')

    # Plotting Accuracy for License Plates and Characters
    width = 0.3  # Width of the bars

    license_bars = axs[2].bar(x - width/2, license_plate_accuracy, width=width, label='License Plate Accuracy')

    character_bars = axs[2].bar(x + width/2, character_accuracy, width=width, label='Character Accuracy')

    axs[2].set_xticks(x)
    axs[2].set_xticklabels(methods)
    axs[2].set_ylabel('Accuracy (%)')
    axs[2].set_title('License Plate and Character Accuracy by Method')
    for bar in license_bars:
        height = bar.get_height()
        axs[2].text(bar.get_x() + bar.get_width() / 2., height + 1, f'{height:.2f}%', 
                    ha='center', va='bottom')

    for bar in character_bars:
        height = bar.get_height()
        axs[2].text(bar.get_x() + bar.get_width() / 2., height + 1, f'{height:.2f}%', 
                    ha='center', va='bottom')

    axs[2].legend()

    plt.tight_layout()
    plt.show()
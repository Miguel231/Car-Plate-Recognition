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

def plot_detection_accuracies(plates_detected, plates_not_detected_previously, total_license_plates,
                              characters_detected, characters_not_detected_previously, total_characters, 
                              total_license_plates_correct, total_characters_correct):
    methods = ['SVM', 'CNN', 'OCR', 'SVM_FILTER', 'CNN_FILTER', 'OCR_FILTER']

    # Calculate accuracies
    license_plate_accuracy = [(detected / total_license_plates) * 100 for detected in plates_detected]
    character_accuracy = [(detected / total_characters) * 100 for detected in characters_detected]
    comb_acc_license = (total_license_plates_correct/total_license_plates)*100
    comb_acc_char = (total_characters_correct/total_characters)*100


    fig, axs = plt.subplots(2, 2, figsize=(12, 12))  # Create a 2x2 subplot
    x = np.arange(len(methods))  # Label locations

    # License Plates Detected Plot
    axs[0, 0].bar(x - 0.2, plates_detected, width=0.4, label='Detected')
    axs[0, 0].bar(x + 0.2, plates_not_detected_previously, width=0.4, label='Not Detected Previously')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(methods)
    axs[0, 0].set_ylabel('Number of License Plates')
    axs[0, 0].set_title('License Plates Detected')
    axs[0, 0].legend()

    # Characters Detected Plot
    axs[0, 1].bar(x - 0.2, characters_detected, width=0.4, label='Detected')
    axs[0, 1].bar(x + 0.2, characters_not_detected_previously, width=0.4, label='Not Detected Previously')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(methods)
    axs[0, 1].set_ylabel('Number of Characters')
    axs[0, 1].set_title('Characters Detected')
    axs[0, 1].legend()

    # Combined Accuracy Plot for License Plates and Characters
    width = 0.35  # Width of the bars
    axs[1, 0].bar(x - width/2, license_plate_accuracy, width=width, label='License Plate Accuracy', alpha=0.7)
    axs[1, 0].bar(x + width/2, character_accuracy, width=width, label='Character Accuracy', alpha=0.7)

    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(methods)
    axs[1, 0].set_ylabel('Accuracy (%)')
    axs[1, 0].set_title('Accuracy of License Plates and Characters')

    # Add accuracy labels on top of the bars for both license plates and characters
    for i in range(len(methods)):
        axs[1, 0].text(i - width/2, license_plate_accuracy[i] + 1, f'{license_plate_accuracy[i]:.2f}%', ha='center')
        axs[1, 0].text(i + width/2, character_accuracy[i] + 1, f'{character_accuracy[i]:.2f}%', ha='center')

    axs[1, 0].legend()

    # Combined Accuracy in the bottom-right subplot (1, 1)
    axs[1, 1].bar(['License Plates', 'Characters'], [comb_acc_license, comb_acc_char], color=['blue', 'orange'])
    axs[1, 1].set_ylim(0, 100)  # Set the limit for y-axis to 0-100%
    axs[1, 1].set_ylabel('Combined Accuracy (%)')
    axs[1, 1].set_title('Combined Accuracy of License Plates and Characters')

    # Add combined accuracy labels on top of the bars
    axs[1, 1].text(0, comb_acc_license + 1, f'{comb_acc_license:.2f}%', ha='center')
    axs[1, 1].text(1, comb_acc_char + 1, f'{comb_acc_char:.2f}%', ha='center')

    plt.tight_layout()
    plt.show()


def plot_train_test_validation(train_accuracies, test_accuracies, val_accuracies, labels):
    x = np.arange(len(labels))  # Label locations
    width = 0.25  # Width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars for each accuracy
    bars1 = ax.bar(x - width, train_accuracies, width, label='Training Accuracy')
    bars2 = ax.bar(x, test_accuracies, width, label='Testing Accuracy')
    bars3 = ax.bar(x + width, val_accuracies, width, label='Validation Accuracy')

    # Add labels and title
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy for License Plates and Characters')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add accuracy labels on top of the bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1, f'{height:.2f}%', 
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
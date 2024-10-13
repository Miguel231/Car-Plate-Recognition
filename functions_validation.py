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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Function to calculate confidence interval
def calculate_ci(p, n, confidence=0.95):
    z = stats.norm.ppf(1 - (1 - confidence) / 2)  # Z-score
    se = np.sqrt((p * (1 - p)) / n)  # Standard error
    ci_lower = p - z * se  # Lower bound
    ci_upper = p + z * se  # Upper bound
    return ci_lower, ci_upper

# Function to calculate confidence intervals and plot them
def calculate_and_plot_confidence_intervals(methods, license_plate_accuracy, character_accuracy, n=100):
    # Check the lengths
    len_license = len(license_plate_accuracy)
    len_character = len(character_accuracy)

    # If lengths are not equal, pad the shorter one with NaNs
    if len_license != len_character:
        max_len = max(len_license, len_character)
        if len_license < max_len:
            license_plate_accuracy += [np.nan] * (max_len - len_license)
        if len_character < max_len:
            character_accuracy += [np.nan] * (max_len - len_character)

    # Combine into a DataFrame for easier analysis
    data = pd.DataFrame({
        'Method': np.repeat(methods, 1),  # Repeat methods for each accuracy
        'License Plate Accuracy': license_plate_accuracy,  # Already padded if necessary
        'Character Accuracy': character_accuracy  # Already padded if necessary
    })

    print(data)

    # Calculate CIs for License Plate Accuracy
    ci_results = []
    for method, accuracy in zip(methods, license_plate_accuracy):
        ci = calculate_ci(accuracy / 100, n)  # Accuracy needs to be a proportion
        ci_results.append((method, 'License Plate Accuracy', accuracy, ci))

    # Calculate CIs for Character Accuracy
    for method, accuracy in zip(methods, character_accuracy):
        ci = calculate_ci(accuracy / 100, n)  # Accuracy needs to be a proportion
        ci_results.append((method, 'Character Accuracy', accuracy, ci))

    # Convert CI results to DataFrame
    ci_df = pd.DataFrame(ci_results, columns=['Method', 'Metric', 'Accuracy', 'CI'])
    ci_df['CI Lower'] = ci_df['CI'].apply(lambda x: x[0])
    ci_df['CI Upper'] = ci_df['CI'].apply(lambda x: x[1])
    ci_df = ci_df.drop(columns='CI')

    # Print CI results
    print("Confidence Intervals for Accuracy Metrics:")
    print(ci_df)

    # Plotting the Confidence Intervals
    plt.figure(figsize=(12, 6))
    for idx, row in ci_df.iterrows():
        plt.errorbar(row['Method'], row['Accuracy'], 
                     yerr=[[row['Accuracy'] - row['CI Lower']], [row['CI Upper'] - row['Accuracy']]], 
                     fmt='o', label=row['Metric'] if idx % 6 == 0 else "", capsize=5)

    plt.title('Confidence Intervals for License Plate and Character Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Function to calculate confidence interval
def calculate_ci(p, n, confidence=0.95):
    z = stats.norm.ppf(1 - (1 - confidence) / 2)  # Z-score
    se = np.sqrt((p * (1 - p)) / n)  # Standard error
    ci_lower = p - z * se  # Lower bound
    ci_upper = p + z * se  # Upper bound
    return ci_lower, ci_upper

# Function to calculate confidence intervals and plot them
def calculate_and_plot_confidence_intervals(methods, license_plate_accuracy, character_accuracy, n=100):
    # Combine into a DataFrame for easier analysis
    data = pd.DataFrame({
        'Method': np.repeat(methods, 1),  # Repeat methods for each accuracy
        'License Plate Accuracy': license_plate_accuracy,  # Already padded if necessary
        'Character Accuracy': character_accuracy  # Already padded if necessary
    })

    # Calculate CIs for License Plate Accuracy
    ci_results = []
    for method, accuracy in zip(methods, license_plate_accuracy):
        ci = calculate_ci(accuracy / 100, n)  # Accuracy needs to be a proportion
        ci_results.append((method, 'License Plate Accuracy', accuracy, ci))

    # Calculate CIs for Character Accuracy
    for method, accuracy in zip(methods, character_accuracy):
        ci = calculate_ci(accuracy / 100, n)  # Accuracy needs to be a proportion
        ci_results.append((method, 'Character Accuracy', accuracy, ci))

    # Convert CI results to DataFrame
    ci_df = pd.DataFrame(ci_results, columns=['Method', 'Metric', 'Accuracy', 'CI'])
    ci_df['CI Lower'] = ci_df['CI'].apply(lambda x: x[0])
    ci_df['CI Upper'] = ci_df['CI'].apply(lambda x: x[1])
    ci_df = ci_df.drop(columns='CI')

    # Display the CI results in a table format
    print("Confidence Intervals for Accuracy Metrics:")
    print(ci_df.to_string(index=False))

    # Plotting the Confidence Intervals
    plt.figure(figsize=(12, 6))
    for idx, row in ci_df.iterrows():
        # Calculate error values ensuring they are non-negative
        lower_error = max(0, row['Accuracy'] - row['CI Lower'])
        upper_error = max(0, row['CI Upper'] - row['Accuracy'])
        
        plt.errorbar(row['Method'], row['Accuracy'], 
                     yerr=[[lower_error], [upper_error]], 
                     fmt='o', label=row['Metric'] if idx % 6 == 0 else "", capsize=5)

    plt.title('Confidence Intervals for License Plate and Character Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()

    # Adding a table below the plot
    plt.table(cellText=ci_df.values,
              colLabels=ci_df.columns,
              cellLoc='center',
              loc='bottom',
              bbox=[0.0, -0.5, 1.0, 0.3])  # Adjust the bbox as needed for your layout

    plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.9, wspace=0.4, hspace=0.7)
    plt.show()
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_ground_truth_from_filenames(image_dir):
    ground_truth = []
    for image_file in os.listdir(image_dir):
        if image_file.endswith('.jpg'):
            label = os.path.splitext(image_file)[0]  
            ground_truth.append(label)
    
    # Sort the list using a custom key
    ground_truth.sort(key=lambda x: (float('inf') if not x.replace('-', '').replace('.', '').isdigit() else float(x), x))
    
    return ground_truth


# Function to load predictions from a .txt file
def load_predictions_from_txt(txt_file):
    predictions = []
    with open(txt_file, 'r') as file:
        for line in file:
            prediction = line.strip()  # Eliminar espacios en blanco alrededor
            prediction = line.strip()  
            if prediction.startswith("E -") or prediction.startswith("E-"):
                prediction = prediction.replace("E -", "").replace("E-", "").strip()

            if len(prediction) > 7:
                prediction = prediction[:7].upper()  
                predictions.append(prediction)
            elif len(prediction) == 7:
                predictions.append(prediction.upper()) 

    return predictions 

def evaluate_predictions(ground_truth, predictions):
    matched_ground_truth = set()  # To keep track of matched ground truth labels
    total_characters_matched = 0
    total_unmatched_characters = 0  # New counter for unmatched characters

    # Define lengths to check in descending order
    lengths_to_check = list(range(7, 2, -1))  # From 7 to 3

    for predicted in predictions:
        matched = False

        # Always check the first ground truth item first
        if ground_truth:
            first_gt = ground_truth[0]  # Get the first ground truth item
            for length in lengths_to_check:
                if len(predicted) >= length:
                    predicted_sequence = predicted[:length]  
                    # Compare against the first ground truth item
                    if first_gt not in matched_ground_truth and predicted_sequence in first_gt:
                        matched_ground_truth.add(first_gt)
                        total_characters_matched += len(predicted_sequence)  
                        matched = True
                        break  # Break once a match is found

        if not matched:  
            total_unmatched_characters += len(predicted)

    print(f"Total Characters Matched: {total_characters_matched}")
    print(f"Total Unmatched Characters: {total_unmatched_characters}")



def run_evaluation_with_filenames(image_dir, svm_txt, cnn_txt, ocr_txt, svm_txt_fil, cnn_txt_fil, ocr_txt_fil):
    ground_truth = load_ground_truth_from_filenames(image_dir)
    svm_predictions = load_predictions_from_txt(svm_txt)
    cnn_predictions = load_predictions_from_txt(cnn_txt)
    ocr_predictions = load_predictions_from_txt(ocr_txt)
    svm_predictions_fil = load_predictions_from_txt(svm_txt_fil)
    cnn_predictions_fil = load_predictions_from_txt(cnn_txt_fil)
    ocr_predictions_fil = load_predictions_from_txt(ocr_txt_fil)
    print("SVM:")
    evaluate_predictions(ground_truth, svm_predictions)
    print("\n")
    print("CNN:")
    evaluate_predictions(ground_truth, cnn_predictions)
    print("\n")
    print("OCR:")    
    evaluate_predictions(ground_truth, ocr_predictions)
    print("\n")
    print("SVM_FILTER:")    
    evaluate_predictions(ground_truth, svm_predictions_fil)
    print("\n")
    print("CNN_FILTER:")
    evaluate_predictions(ground_truth, cnn_predictions_fil)
    print("\n")
    print("OCR_FILTER:")    
    evaluate_predictions(ground_truth, ocr_predictions_fil)


    
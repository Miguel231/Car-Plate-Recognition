import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def print_evaluation_metrics(true_labels, predictions, model_name, class_names):
    print(f"Evaluation Metrics for {model_name}:")
    report = classification_report(true_labels, predictions, target_names=class_names)
    print(report)

def plot_confusion_matrix(true_labels, predictions, title):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

def plot_metrics_comparison(cnn_metrics, svm_metrics, ocr_metrics,cnn_metrics_fil, svm_metrics_fil, ocr_metrics_fil,metric_names):
    x = np.arange(len(metric_names)) 
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width, cnn_metrics, width, label='CNN')
    ax.bar(x, svm_metrics, width, label='SVM')
    ax.bar(x + width, ocr_metrics, width, label='OCR')

    ax.bar(x - width, cnn_metrics_fil, width, label='CNN_F', alpha=0.5)
    ax.bar(x, svm_metrics_fil, width, label='SVM_F', alpha=0.5)
    ax.bar(x + width, ocr_metrics_fil, width, label='OCR_F', alpha=0.5)

    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Metrics between CNN, SVM and OCR')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()

    plt.show()

import os

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
            # Si la línea está vacía, agregar "U" a las predicciones
            if not prediction:
                predictions.append("U")
                continue
            prediction = line.strip()  
            if prediction.startswith("E -") or prediction.startswith("E-"):
                prediction = prediction.replace("E -", "").replace("E-", "").strip()

            if len(prediction) > 7:
                prediction = prediction[:7].upper()  
                predictions.append(prediction)
            elif len(prediction) == 7:
                predictions.append(prediction.upper()) 

    return predictions 

def evaluate_predictions(ground_truth, predictions, model_name):
    matched_ground_truth = set()  # To keep track of matched ground truth labels
    total_correct = 0
    total_characters_matched = 0
    total_mismatched = 0
    total_unmatched_characters = 0  # New counter for unmatched characters

    # Define lengths to check in descending order
    lengths_to_check = list(range(7, 2, -1))  # From 7 to 3

    for predicted in predictions:
        matched = False
        # Check for sequences of various lengths
        for length in lengths_to_check:
            # Check if the predicted sequence is long enough
            if len(predicted) >= length:
                # Extract the substring to check
                predicted_sequence = predicted[:length]  # Take the first 'length' characters
                
                # Look for a match in ground truth
                for gt in ground_truth:
                    if gt not in matched_ground_truth and predicted_sequence in gt:
                        # If a match is found, mark it
                        matched_ground_truth.add(gt)
                        total_correct += 1
                        total_characters_matched += len(predicted_sequence)  # Add length of matched sequence
                        matched = True
                        break  # Exit the ground truth loop once matched

            if matched:  # If a match was found, break out of the length loop
                break

        if not matched:  # If no match was found
            total_mismatched += 1
            total_unmatched_characters += len(predicted)  # Add length of unmatched predicted sequence

    print(f"Total Correct Matches: {total_correct}")
    print(f"Total Characters Matched: {total_characters_matched}")
    print(f"Total Mismatched Predictions: {total_mismatched}")
    print(f"Total Unmatched Characters: {total_unmatched_characters}")

    #plot_confusion_matrix(ground_truth, predictions, model_name)


def run_evaluation_with_filenames(image_dir, svm_txt, cnn_txt, ocr_txt, svm_txt_fil, cnn_txt_fil, ocr_txt_fil):
    ground_truth = load_ground_truth_from_filenames(image_dir)
    svm_predictions = load_predictions_from_txt(svm_txt)
    cnn_predictions = load_predictions_from_txt(cnn_txt)
    ocr_predictions = load_predictions_from_txt(ocr_txt)
    svm_predictions_fil = load_predictions_from_txt(svm_txt_fil)
    cnn_predictions_fil = load_predictions_from_txt(cnn_txt_fil)
    ocr_predictions_fil = load_predictions_from_txt(ocr_txt_fil)
    evaluate_predictions(ground_truth, svm_predictions, "SVM")
    evaluate_predictions(ground_truth, cnn_predictions, "CNN")
    evaluate_predictions(ground_truth, ocr_predictions, "OCR")
    evaluate_predictions(ground_truth, svm_predictions_fil, "SVM_F")
    evaluate_predictions(ground_truth, cnn_predictions_fil, "CNN_F")
    evaluate_predictions(ground_truth, ocr_predictions_fil, "OCR_f")

    svm_accuracy = np.mean([gt == pred for gt, pred in zip(ground_truth, svm_predictions)])
    cnn_accuracy = np.mean([gt == pred for gt, pred in zip(ground_truth, cnn_predictions)])
    ocr_accuracy = np.mean([gt == pred for gt, pred in zip(ground_truth, ocr_predictions)])
    svm_accuracy_fil = np.mean([gt == pred for gt, pred in zip(ground_truth, svm_predictions_fil)])
    cnn_accuracy_fil = np.mean([gt == pred for gt, pred in zip(ground_truth, cnn_predictions_fil)])
    ocr_accuracy_fil = np.mean([gt == pred for gt, pred in zip(ground_truth, ocr_predictions_fil)])

    metric_names = ['Accuracy']
    cnn_metrics = [cnn_accuracy]
    svm_metrics = [svm_accuracy]
    ocr_metrics = [ocr_accuracy]
    cnn_metrics_fil = [cnn_accuracy_fil]
    svm_metrics_fil = [svm_accuracy_fil]
    ocr_metrics_fil = [ocr_accuracy_fil]

    plot_metrics_comparison(cnn_metrics, svm_metrics, ocr_metrics,cnn_metrics_fil, svm_metrics_fil, ocr_metrics_fil, metric_names)


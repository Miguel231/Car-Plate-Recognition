import os
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def print_evaluation_metrics(true_labels, predictions, model_name, class_names):
    print(f"Evaluation Metrics for {model_name}:")
    report = classification_report(true_labels, predictions, target_names=class_names)
    print(report)

def plot_confusion_matrix(true_labels, predictions, title, class_names):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

def plot_metrics_comparison(cnn_metrics, svm_metrics, ocr_metrics,metric_names):
    metrics_cnn, metrics_svm, metrics_ocr = cnn_metrics, svm_metrics, ocr_metrics
    x = np.arange(len(metric_names))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, metrics_cnn, width, label='CNN')
    rects2 = ax.bar(x + width/2, metrics_svm, width, label='SVM')
    rects3 = ax.bar(x + width/2, metrics_ocr, width, label='OCR')

    # Add labels, title, and custom x-axis tick labels
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Metrics between CNN and SVM')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()

    plt.show()


# Function to load ground truth labels from image filenames
def load_ground_truth_from_filenames(image_dir):
    ground_truth = []
    for image_file in os.listdir(image_dir):
        if image_file.endswith('.jpg'):
            label = os.path.splitext(image_file)[0]  
            ground_truth.append(label)
    return ground_truth

# Function to load predictions from a .txt file
def load_predictions_from_txt(txt_file):
    predictions = []
    with open(txt_file, 'r') as file:
        for line in file:
            prediction = line.strip()  

            if prediction.startswith("E -") or prediction.startswith("E-"):
                prediction = prediction.replace("E -", "").replace("E-", "").strip()

            if len(prediction) >= 7:
                prediction = prediction[:7].upper()  
                print(prediction)
                predictions.append(prediction)
            elif len(prediction) == 7:
                predictions.append(prediction.upper()) 
    
    return predictions



def evaluate_predictions(ground_truth, predictions, model_name, class_names):
    print_evaluation_metrics(ground_truth, predictions, model_name, class_names)
    plot_confusion_matrix(ground_truth, predictions, model_name, class_names)


def run_evaluation_with_filenames(image_dir, svm_txt, cnn_txt, ocr_txt, class_names):
    ground_truth = load_ground_truth_from_filenames(image_dir)
    svm_predictions = load_predictions_from_txt(svm_txt)
    cnn_predictions = load_predictions_from_txt(cnn_txt)
    ocr_predictions = load_predictions_from_txt(ocr_txt)
    evaluate_predictions(ground_truth, svm_predictions, "SVM", class_names)
    evaluate_predictions(ground_truth, cnn_predictions, "CNN", class_names)
    evaluate_predictions(ground_truth, ocr_predictions, "OCR", class_names)

    svm_accuracy = np.mean([gt == pred for gt, pred in zip(ground_truth, svm_predictions)])
    cnn_accuracy = np.mean([gt == pred for gt, pred in zip(ground_truth, cnn_predictions)])
    ocr_accuracy = np.mean([gt == pred for gt, pred in zip(ground_truth, ocr_predictions)])

    metric_names = ['Accuracy']
    cnn_metrics = [cnn_accuracy]
    svm_metrics = [svm_accuracy]
    ocr_metrics = [ocr_accuracy]

    plot_metrics_comparison(cnn_metrics, svm_metrics, ocr_metrics, metric_names)


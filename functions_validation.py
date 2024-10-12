import os
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

def evaluate_predictions(ground_truth, predictions, model_name):
    gt_length = len(ground_truth)
    pred_length = len(predictions)

    # Manejo de longitudes desiguales
    if gt_length != pred_length:
        print(f"Length mismatch for {model_name}:")
        print(f"Ground Truth length: {gt_length}")
        print(f"Predictions length: {pred_length}")

        # Ajustar a la longitud más corta
        min_length = min(gt_length, pred_length)
        ground_truth = ground_truth[:min_length]
        predictions = predictions[:min_length]

        print(f"Adjusted Ground Truth length: {len(ground_truth)}")
        print(f"Adjusted Predictions length: {len(predictions)}")

    # Clases únicas en las predicciones y ground truth
    unique_gt_classes = set(ground_truth)
    unique_pred_classes = set(predictions)

    print(f"Unique classes in Ground Truth: {unique_gt_classes}")
    print(f"Unique classes in Predictions: {unique_pred_classes}")

    # Contadores para evaluaciones
    true_counts = []
    false_counts = []

    for gt, pred in zip(ground_truth, predictions):
        # Identificación de coincidencias
        match_length = sum(1 for a, b in zip(gt, pred) if a == b)
        true_counts.append(match_length)
        false_counts.append(len(gt) - match_length)

    total_true = sum(true_counts)
    total_false = sum(false_counts)

    print(f"Total Correct Identifications: {total_true}, Total Misidentifications: {total_false}")

    plot_confusion_matrix(ground_truth, predictions, model_name)


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


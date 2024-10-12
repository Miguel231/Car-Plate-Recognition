import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to evaluate the CNN model
def evaluate_cnn_model(cnn_model, test_loader, criterion):
    cnn_model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = cnn_model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Track predictions and actual labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Calculate the loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Calculate accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = running_loss / len(test_loader)
    
    return all_labels, all_preds, accuracy, avg_loss

from sklearn.svm import SVC

# Function to evaluate the SVM model
def evaluate_svm_model(svm_model, test_features, test_labels):
    # Predict using SVM
    svm_preds = svm_model.predict(test_features)
    
    # Calculate metrics
    accuracy = np.mean(svm_preds == test_labels)
    
    return test_labels, svm_preds, accuracy

from sklearn.metrics import classification_report

# Function to print evaluation metrics
def print_evaluation_metrics(true_labels, predictions, model_name):
    print(f"Evaluation Metrics for {model_name}:")
    report = classification_report(true_labels, predictions, target_names=class_names)
    print(report)

# Function to plot confusion matrix
def plot_confusion_matrix(true_labels, predictions, title):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

def plot_metrics_comparison(cnn_metrics, svm_metrics, metric_names):
    metrics_cnn, metrics_svm = cnn_metrics, svm_metrics
    x = np.arange(len(metric_names))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, metrics_cnn, width, label='CNN')
    rects2 = ax.bar(x + width/2, metrics_svm, width, label='SVM')

    # Add labels, title, and custom x-axis tick labels
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Metrics between CNN and SVM')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()

    plt.show()

import pandas as pd

# Function to generate a summary table
def generate_summary_table(cnn_metrics, svm_metrics, metric_names):
    summary_data = {
        'Metric': metric_names,
        'CNN': cnn_metrics,
        'SVM': svm_metrics
    }
    
    df = pd.DataFrame(summary_data)
    print(df)

def run_full_evaluation(cnn_model, svm_model, test_loader, test_features, test_labels, class_names):
    criterion = nn.CrossEntropyLoss()

    # Evaluate CNN Model
    cnn_true_labels, cnn_preds, cnn_accuracy, cnn_loss = evaluate_cnn_model(cnn_model, test_loader, criterion)
    print_evaluation_metrics(cnn_true_labels, cnn_preds, "CNN")
    plot_confusion_matrix(cnn_true_labels, cnn_preds, "CNN License Plate Prediction")
    
    # Evaluate SVM Model
    svm_true_labels, svm_preds, svm_accuracy = evaluate_svm_model(svm_model, test_features, test_labels)
    print_evaluation_metrics(svm_true_labels, svm_preds, "SVM")
    plot_confusion_matrix(svm_true_labels, svm_preds, "SVM License Plate Prediction")
    
    # Comparing Metrics
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    cnn_metrics = [cnn_accuracy]  # Add precision, recall, F1-score to this list after calculating them
    svm_metrics = [svm_accuracy]  # Add precision, recall, F1-score to this list after calculating them
    
    plot_metrics_comparison(cnn_metrics, svm_metrics, metric_names)
    generate_summary_table(cnn_metrics, svm_metrics, metric_names)
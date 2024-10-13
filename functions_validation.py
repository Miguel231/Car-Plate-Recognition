import os
import matplotlib.pyplot as plt


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
            if prediction.startswith("E -") or prediction.startswith("E-"):
                prediction = prediction.replace("E -", "").replace("E-", "").strip()
                
            if len(prediction) > 7:
                prediction = prediction[:7].upper()  
            elif len(prediction) == 7:
                prediction = prediction.upper() 

            predictions.append(prediction)

    return predictions 


def evaluate_predictions(ground_truth, predictions, model_name):
    total_correct = 0
    total_characters_matched = 0
    total_unmatched_characters = 0  # New counter for unmatched characters

    for gt, pred in zip(ground_truth, predictions):
        if gt == pred:
            total_correct += 1  # Count total correct predictions
            total_characters_matched += len(gt)  # Count matched characters
        else:
            # Count unmatched characters if they don't match
            total_unmatched_characters += len(pred)

        # Additional character-by-character comparison (optional)
        for g_char, p_char in zip(gt, pred):
            if g_char == p_char:
                total_characters_matched += 1

    print(f"Model: {model_name}")
    print(f"Total Correct Predictions: {total_correct} out of {len(ground_truth)}")
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


    
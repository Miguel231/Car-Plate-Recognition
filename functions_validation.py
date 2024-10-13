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
            if prediction.startswith("E -") or prediction.startswith("E-") or prediction.startswith("E - ") or prediction.startswith("E- ") or prediction.startswith("€") :
                prediction = prediction.replace("E -", "").replace("E-", "").replace("E - ", "").replace("E- ", "").strip()
                
            if len(prediction) > 7:
                prediction = prediction.upper()  
            elif len(prediction) == 7:
                prediction = prediction.upper() 

            predictions.append(prediction)

    return predictions 


def evaluate_predictions(ground_truth, predictions, f, num):
    #total_characters_matched = 0
    #total_unmatched_characters = 0  # New counter for unmatched characters
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
    #print(f"Total Characters Matched: {total_characters_matched}")
    #print(f"Total Unmatched Characters: {total_unmatched_characters}")
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
    num = 0
    print("SVM:")
    num = evaluate_predictions(ground_truth, svm_predictions, f, num)
    print("\n")
    print("CNN:")
    evaluate_predictions(ground_truth, cnn_predictions, f, num)
    print("\n")
    print("OCR:")    
    evaluate_predictions(ground_truth, ocr_predictions, f, num)
    print("\n")
    print("SVM_FILTER:")    
    evaluate_predictions(ground_truth, svm_predictions_fil, f, num)
    print("\n")
    print("CNN_FILTER:")
    evaluate_predictions(ground_truth, cnn_predictions_fil, f, num)
    print("\n")
    print("OCR_FILTER:")    
    f, num = evaluate_predictions(ground_truth, ocr_predictions_fil, f, num)
    s = 0
    for values in f:
        if values == 1:
            s+=1
    print("TOTAL DETECTED CORRECTLY: ", s)




    
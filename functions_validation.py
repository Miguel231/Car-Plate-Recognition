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
    print("TOTAL:",c)
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
    print("TOTAL:",c)
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
    print("TOTAL LICENSE PLATES DETECTED CORRECTLY: ", suma)   
    print("\n")
    print("\n") 


    ### BY CHARACTERS ##################################
    suma = 0
    num = 0
    print("CHARACTERS DETECTED: ")
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
    print("TOTAL CHARACTERS DETECTED CORRECTLY: ", suma)   


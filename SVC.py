import matplotlib.pyplot as plt
import os
import numpy as np
import os
from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,  confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from PIL import Image

def train_svm_and_get_accuracy(base_path, class_names):
    def load_images_from_folder(base_path):
            images = []
            labels = []

            valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

            #through each folder (A-Z, 0-9)
            for folder in os.listdir(base_path):
                folder_path = os.path.join(base_path, folder)
                if os.path.isdir(folder_path):
                    for img_name in os.listdir(folder_path):
                        img_path = os.path.join(folder_path, img_name)

                        _, ext = os.path.splitext(img_name)
                        if ext.lower() in valid_image_extensions:
                            try:
                                img = Image.open(img_path).convert('L').resize((28, 28))
                                img_array = np.array(img).flatten()  # Flatten the image
                                images.append(img_array)

                                #folder name as label
                                labels.append(folder) 
                            except Exception as e:
                                print(f"Error loading image {img_name}: {e}")
                                continue  

            return np.array(images), np.array(labels)


    images, labels = load_images_from_folder(base_path)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, encoded_labels, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    train_images = scaler.fit_transform(train_images)
    test_images = scaler.transform(test_images)

    clf = svm.SVC(kernel='linear', probability=True)  # Linear kernel
    clf.fit(train_images, train_labels)

    test_predictions = clf.predict(test_images)

    accuracy = accuracy_score(test_labels, test_predictions)

    conf_matrix = confusion_matrix(test_labels, test_predictions, labels=np.arange(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    
    return accuracy, clf, scaler, label_encoder

def test_preprocessed_images_with_plot(clf, scaler, character_list, label_encoder):
    plt.figure(figsize=(10, 10))  
    plate = ''  

    for i, char_image in enumerate((character_list)):
        
        img_resized = Image.fromarray(char_image).resize((28, 28))
        img_array = np.array(img_resized).flatten()

        img_array_scaled = scaler.transform([img_array])
        ###################################################################
        probs = clf.predict_proba(img_array_scaled)[0]  
        max_prob = max(probs)
        print(max_prob)

        if max_prob < 0.095:
            plate += '?'  
            continue  
        
        predicted_label_index = np.argmax(probs)
        predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
        #######################################################################

        """
        #prediction using the trained SVM model
        predicted_label_index = clf.predict(img_array_scaled)[0]
        predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
        """
        #save the changes
        #output_path = os.path.join("/content/drive/MyDrive/LICENSE_PLATES_RECOGITION_L&V/Dataset Characters/", f'{predicted_label}/caracter_{i}.png')
        #cv2.imwrite(output_path, characters[i])

        plate = plate + predicted_label
    return plate
import matplotlib.pyplot as plt
import os
import numpy as np
import os
from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from PIL import Image


def train_svm_and_get_accuracy(base_path):
    """
    Loads images from the specified folder, trains an SVM classifier, and returns the accuracy.

    Parameters:
    - base_path (str): The base directory containing subdirectories of labeled images.

    Returns:
    - accuracy (float): The accuracy of the trained SVM classifier on the test set.
    """
    def load_images_from_folder(base_path):
            images = []
            labels = []

            # Valid image extensions
            valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

            # Iterate through each folder (A-Z, 0-9)
            for folder in os.listdir(base_path):
                folder_path = os.path.join(base_path, folder)
                if os.path.isdir(folder_path):
                    # Inside the folder, load each image
                    for img_name in os.listdir(folder_path):
                        img_path = os.path.join(folder_path, img_name)

                        # Get file extension and check if it's a valid image file
                        _, ext = os.path.splitext(img_name)
                        if ext.lower() in valid_image_extensions:
                            try:
                                # Load the image, convert to grayscale, and resize to a standard size (e.g., 28x28)
                                img = Image.open(img_path).convert('L').resize((28, 28))
                                img_array = np.array(img).flatten()  # Flatten the image
                                images.append(img_array)

                                # Assign the folder name as label
                                labels.append(folder)  # Keep folder names as labels
                            except Exception as e:
                                print(f"Error loading image {img_name}: {e}")
                                continue  # Skip the problematic image

            return np.array(images), np.array(labels)


    # Load the images and labels
    images, labels = load_images_from_folder(base_path)

    # Encode string labels into integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Split the dataset into training and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, encoded_labels, test_size=0.2, random_state=42
    )

    # Scale the images (important for SVM)
    scaler = StandardScaler()
    train_images = scaler.fit_transform(train_images)
    test_images = scaler.transform(test_images)

    # Train the SVM classifier
    clf = svm.SVC(kernel='linear')  # Linear kernel
    clf.fit(train_images, train_labels)

    # Predict on the test set
    test_predictions = clf.predict(test_images)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(test_labels, test_predictions)
    
    return accuracy, clf, scaler, label_encoder

def test_preprocessed_images_with_plot(clf, scaler, character_list, label_encoder):
    plt.figure(figsize=(10, 10))  # Set the figure size
    plate = ''  # This will hold the concatenated predicted characters

    # Reverse the order of the character_list and iterate over it
    for i, char_image in enumerate((character_list)):
        # Resize the image to 28x28 if necessary
        img_resized = Image.fromarray(char_image).resize((28, 28))

        # Convert resized image back to a NumPy array and flatten it
        img_array = np.array(img_resized).flatten()

        # Scale the image using the trained scaler
        img_array_scaled = scaler.transform([img_array])

        # Make a prediction using the trained SVM model
        predicted_label_index = clf.predict(img_array_scaled)[0]

        # Convert the predicted label index back to the original label
        predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]

        #save the changes
        #output_path = os.path.join("/content/drive/MyDrive/LICENSE_PLATES_RECOGITION_L&V/Dataset Characters/", f'{predicted_label}/caracter_{i}.png')
        #cv2.imwrite(output_path, characters[i])

        plate = plate + predicted_label
    return plate
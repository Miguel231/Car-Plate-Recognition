import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.view(-1, 128 * 3 * 3)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    print("Finished Training")


def evaluate_model(model, test_loader, class_names):
    model.eval()  
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")

    cm = confusion_matrix(all_labels, all_predictions, labels=np.arange(len(class_names)))
    
    # Display confusion matrix with class names
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=90)  # Rotate labels for readability
    plt.show()

def preprocess_character_image(char_image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  
        transforms.Resize((28, 28)),                  
        transforms.ToTensor(),                    
        transforms.Normalize((0.5,), (0.5,))          
    ])
    
    char_image = Image.fromarray(char_image)  
    return transform(char_image).unsqueeze(0)  # add dimension for batch

def predict_characters(model, character_list, label_encoder):
    model.eval() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    plate = ''  # save predicted plate
    for i, char_image in enumerate(character_list):
        preprocessed_image = preprocess_character_image(char_image).to(device)

        with torch.no_grad():  
            outputs = model(preprocessed_image)  
            _, predicted_index = torch.max(outputs, 1)  
            predicted_label = label_encoder.inverse_transform([predicted_index.item()])[0]  
            plate += predicted_label

    return plate
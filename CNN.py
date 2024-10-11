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

# Configurar dispositivo (GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        # Capa convolucional 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Capa convolucional 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Capa convolucional 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Capa completamente conectada
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Paso por las capas convolucionales + pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        # Aplanar el tensor
        x = x.view(-1, 128 * 3 * 3)

        # Pasar por capas completamente conectadas
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# Función para entrenar el modelo
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Inicializar los gradientes a cero
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass y optimización
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    print("Finished Training")


# Evaluar el modelo (opcionalmente con un dataset de prueba)
def evaluate_model(model, test_loader):
    model.eval()  # Cambiar el modelo al modo de evaluación
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")

# Función para preprocesar la imagen de un carácter
def preprocess_character_image(char_image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convertir a escala de grises
        transforms.Resize((28, 28)),                  # Redimensionar a 28x28 píxeles
        transforms.ToTensor(),                        # Convertir a tensor
        transforms.Normalize((0.5,), (0.5,))          # Normalizar entre -1 y 1
    ])
    
    char_image = Image.fromarray(char_image)  # Convertir array NumPy a imagen PIL
    return transform(char_image).unsqueeze(0)  # Añadir una dimensión extra para el batch

# Función para predecir los caracteres en una lista de imágenes de caracteres
def predict_characters(model, character_list, label_encoder):
    model.eval()  # Cambiar a modo de evaluación
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    plate = ''  # Guardar la matrícula predicha

    for i, char_image in enumerate(character_list):
        # Preprocesar cada imagen de carácter
        preprocessed_image = preprocess_character_image(char_image).to(device)

        with torch.no_grad():  # No necesitamos gradientes en evaluación
            outputs = model(preprocessed_image)  # Pasar la imagen por el modelo
            _, predicted_index = torch.max(outputs, 1)  # Obtener la predicción
            predicted_label = label_encoder.inverse_transform([predicted_index.item()])[0]  # Decodificar la predicción

            # Agregar el carácter predicho a la cadena de la matrícula
            plate += predicted_label

    return plate




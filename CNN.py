import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

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

# Llamar a la función de evaluación (asume que tienes un `test_loader`)
# evaluate_model(model, test_loader)

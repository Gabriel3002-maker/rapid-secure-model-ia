import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
import os

# Definir la clase del modelo
class AccidentDetectionModel(nn.Module):
    def __init__(self):
        super(AccidentDetectionModel, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Dos clases

    def forward(self, x):
        return self.model(x)

# Cargar el modelo entrenado
model = AccidentDetectionModel()
model.load_state_dict(torch.load('./modelo_entrenado.pth'))
model.eval()

# Cargar tus datos de prueba
imagenes_test_tensor = torch.load('./imagenes_tensor.pt')
etiquetas_test_tensor = torch.load('./etiquetas_tensor.pt')

# Crear DataLoader para el conjunto de prueba
test_dataset = TensorDataset(imagenes_test_tensor, etiquetas_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluar el modelo
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Precisión del modelo: {100 * correct / total:.2f}%')

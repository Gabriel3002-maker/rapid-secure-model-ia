import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset

# Cargar los tensores guardados
imagenes_tensor = torch.load('imagenes_tensor.pt')
etiquetas_tensor = torch.load('etiquetas_tensor.pt')

# Crear un DataLoader
dataset = TensorDataset(imagenes_tensor, etiquetas_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Definir el modelo
class AccidentDetectionModel(nn.Module):
    def __init__(self):
        super(AccidentDetectionModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Dos clases

    def forward(self, x):
        return self.model(x)

# Crear una instancia del modelo
model = AccidentDetectionModel()

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento
num_epochs = 10  # Cambia según sea necesario

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')

torch.save(model.state_dict(), 'modelo_entrenado.pth')
print("Modelo guardado como 'modelo_entrenado.pth'")

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

# Definir tu modelo aquí
class AccidentDetectionModel(nn.Module):
    def __init__(self):
        super(AccidentDetectionModel, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Dos clases

    def forward(self, x):
        return self.model(x)

# Cargar el modelo entrenado
model = AccidentDetectionModel()
model.load_state_dict(torch.load('modelo_entrenado.pth'))
model.eval()

# Función de predicción
def predecir(imagen):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    imagen_transformada = transform(imagen).unsqueeze(0)

    with torch.no_grad():
        output = model(imagen_transformada)
        _, predicted = torch.max(output.data, 1)

    return 'Accidente' if predicted.item() == 1 else 'No Accidente'

# Captura de video
cap = cv2.VideoCapture(0)  # 0 para la cámara por defecto

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Hacer la predicción
    resultado = predecir(frame)

    # Mostrar el resultado en el video
    cv2.putText(frame, resultado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Detección de Emergencias', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

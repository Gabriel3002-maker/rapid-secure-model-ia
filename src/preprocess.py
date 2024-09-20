import os
import cv2
import torch
from torchvision import transforms

# Rutas a las carpetas
ruta_accidentes = os.path.expanduser('~/Documents/rapid-secure-model-ia/data/accident_traffic/train/Accident')
ruta_no_accidentes = os.path.expanduser('~/Documents/rapid-secure-model-ia/data/accident_traffic/train/Non Accident')

# Transformaciones: redimensionar y convertir a tensor
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Listas para almacenar imágenes y etiquetas
imagenes = []
etiquetas = []

# Cargar imágenes de accidentes
for archivo in os.listdir(ruta_accidentes):
    if archivo.endswith('.jpg') or archivo.endswith('.png'):
        imagen_path = os.path.join(ruta_accidentes, archivo)
        imagen = cv2.imread(imagen_path)
        imagen_transformada = transform(imagen)
        imagenes.append(imagen_transformada)
        etiquetas.append(1)  # 1 para "accidente"

# Cargar imágenes de no accidentes
for archivo in os.listdir(ruta_no_accidentes):
    if archivo.endswith('.jpg') or archivo.endswith('.png'):
        imagen_path = os.path.join(ruta_no_accidentes, archivo)
        imagen = cv2.imread(imagen_path)
        imagen_transformada = transform(imagen)
        imagenes.append(imagen_transformada)
        etiquetas.append(0)  # 0 para "no accidente"

# Convertir listas a tensores
imagenes_tensor = torch.stack(imagenes)
etiquetas_tensor = torch.tensor(etiquetas)

print("Imágenes y etiquetas cargadas y preprocesadas.")

# Guardar los tensores
torch.save(imagenes_tensor, 'imagenes_tensor.pt')
torch.save(etiquetas_tensor, 'etiquetas_tensor.pt')

print("Tensores guardados.")

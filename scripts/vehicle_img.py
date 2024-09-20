import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Configuración
search_query = "car crash"
num_images = 10  # Número de imágenes a descargar
save_dir = "imagenes_chocques"

# Crear el directorio si no existe
os.makedirs(save_dir, exist_ok=True)

# Realizar la búsqueda en Google Imágenes
url = f"https://www.google.com/search?hl=en&tbm=isch&q={search_query}"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# Encontrar las imágenes
image_elements = soup.find_all('img', limit=num_images + 1)  # +1 porque la primera imagen es el logo

# Descargar las imágenes
for i, img in enumerate(image_elements[1:], start=1):  # Ignorar la primera imagen
    img_url = img['src']
    try:
        img_data = requests.get(img_url).content
        with open(os.path.join(save_dir, f"car_crash_{i}.jpg"), 'wb') as handler:
            handler.write(img_data)
        print(f"Descargada: car_crash_{i}.jpg")
    except Exception as e:
        print(f"No se pudo descargar la imagen {i}: {e}")

print("Descarga completa.")

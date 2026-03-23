import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import time

# --- 1. CONFIGURACIÓN ---
ESP32_IP = "192.168.1.27"
URL = f"http://{ESP32_IP}/"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELO_PATH = 'garbage_model_binario_local.pth'

# --- 2. PREPARAR IA ---
def preparar_modelo():
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )
    if os.path.exists(MODELO_PATH):
        model.load_state_dict(torch.load(MODELO_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    return None

model_ia = preparar_modelo()

transformaciones = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. CAPTURA ESTABLE ---
def obtener_foto_estable():
    try:
        response = requests.get(URL, timeout=15, stream=True)
        if response.status_code == 200:
            image_buffer = io.BytesIO()
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    image_buffer.write(chunk)
            image_buffer.seek(0)
            return Image.open(image_buffer).convert('RGB')
    except Exception as e:
        print(f"❌ Error de captura: {e}")
    return None

# --- 4. BUCLE PRINCIPAL CON RETORNO RÁPIDO (5s) ---
def ejecutar_sistema():
    print("\n🚀 SISTEMA UDI - CONTROL DE CICLO CORTO (5s)")
    print("Estado inicial: Motor en 0° (Reposos)")
    
    try:
        while True:
            input("\n📸 [ENTER] para iniciar nueva clasificación...")
            
            img = obtener_foto_estable()
            
            if img:
                img_t = transformaciones(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    salida = model_ia(img_t)
                    prob = torch.nn.functional.softmax(salida, dim=1)
                    conf, pred = torch.max(prob, 1)

                resultado = pred.item()
                if resultado == 0 and conf.item() < 0.90:
                    resultado = 1 

                # --- ACCIÓN DEL MOTOR ---
                print("\n" + "="*50)
                if resultado == 0:
                    print(f"🟢 APROVECHABLE ({conf.item()*100:.1f}%)")
                    print("⚙️  MOTOR: Girando a +90° (Apertura Derecha)")
                else:
                    print(f"🔴 NO APROVECHABLE ({confianza.item()*100:.1f}%)")
                    print("⚙️  MOTOR: Girando a -90° (Apertura Izquierda)")
                print("="*50)

                # --- TEMPORIZADOR DE 5 SEGUNDOS ---
                print(f"⏳ Procesando descarga (5 segundos)...")
                for i in range(5, 0, -1):
                    print(f"  Cerrando en: {i}s  ", end="\r")
                    time.sleep(1)
                
                print("\n🔄 MOTOR: Volviendo a POSICIÓN INICIAL (0°)")
                print("✅ Listo para el siguiente residuo.")

    except KeyboardInterrupt:
        print("\n👋 Programa finalizado.")

if __name__ == '__main__':
    ejecutar_sistema()
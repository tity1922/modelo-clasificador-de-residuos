import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')

def cargar_modelo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'garbage_model_binario_local.pth'
    
    # Arquitectura idéntica al entrenamiento v3.1
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model, device
    return None, None

def ejecutar_interfaz():
    model, device = cargar_modelo()
    if not model:
        print("❌ Error: No se encontró el modelo entrenado.")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    while True:
        limpiar_pantalla()
        print("==================================================")
        print("  ♻️  SISTEMA DE CLASIFICACIÓN DE RESIDUOS - UIS  ")
        print("==================================================")
        print(" 1. Arrastra la foto o pega la ruta.")
        print(" 2. Escribe 'salir' para cerrar.")
        print("--------------------------------------------------")

        ruta_sucia = input("\n📥 Ruta de la imagen: ").strip()
        if ruta_sucia.lower() in ['salir', 's']: break
        
        ruta_limpia = ruta_sucia.replace('"', '').replace("'", "")

        if os.path.exists(ruta_limpia):
            try:
                img = Image.open(ruta_limpia).convert('RGB')
                img_t = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(img_t)
                    prob = torch.nn.functional.softmax(output, dim=1)
                    conf, pred = torch.max(prob, 1)

                # --- LÓGICA DE SEGURIDAD (FALLBACK) ---
                umbral_seguridad = 0.90  # 90% de confianza mínima
                
                # Si es 'Aprovechable' pero con baja confianza, forzamos a 'No Aprovechable'
                # Esto captura fondos vacíos o ruidosos.
                if pred.item() == 0 and conf.item() < umbral_seguridad:
                    resultado_final = 1 
                    estado_tecnico = f"⚠️ RECLASIFICADO POR INCERTIDUMBRE ({conf.item()*100:.1f}%)"
                else:
                    resultado_final = pred.item()
                    estado_tecnico = f"✅ CLASIFICACIÓN SEGURA ({conf.item()*100:.1f}%)"

                categorias = ['APROVECHABLE ✅', 'NO APROVECHABLE ❌']
                color = "\033[92m" if resultado_final == 0 else "\033[91m"
                reset = "\033[0m"

                print(f"\n🔍 REPORTE TÉCNICO: {estado_tecnico}")
                print(f"--------------------------------------------------")
                print(f"RESULTADO FINAL: {color}{categorias[resultado_final]}{reset}")
                print(f"--------------------------------------------------")
                input("\nPresiona ENTER para continuar...")

            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("\nPresiona ENTER...")
        else:
            print(f"\n⚠️ Archivo no encontrado.")
            input("\nPresiona ENTER...")

if __name__ == '__main__':
    ejecutar_interfaz()
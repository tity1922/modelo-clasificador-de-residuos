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
    
    # 1. Reconstruir la arquitectura EXACTA del nuevo entrenamiento (v3.1)
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    
    # IMPORTANTE: Esta estructura debe ser idéntica a la del main.py
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)  # 2 clases
    )
    
    # 2. Cargar los pesos entrenados
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            return model, device
        except Exception as e:
            print(f"❌ Error interno al cargar los pesos: {e}")
            return None, None
    return None, None

def ejecutar_interfaz():
    model, device = cargar_modelo()
    if not model:
        print("❌ Error: No se encontró el modelo .pth o la arquitectura no coincide.")
        return

    # Transformaciones estándar de ImageNet
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
        print(" Instrucciones:")
        print(" 1. Busca tu foto en el explorador de Windows.")
        print(" 2. Arrástrala directamente aquí o pega la ruta.")
        print(" 3. Escribe 'salir' para cerrar.")
        print("--------------------------------------------------")

        ruta_sucia = input("\n📥 Ruta de la imagen: ").strip()

        if ruta_sucia.lower() in ['salir', 's', 'exit']:
            break
        
        # Limpiar comillas que pone Windows al "Copiar como ruta"
        ruta_limpia = ruta_sucia.replace('"', '').replace("'", "")

        if os.path.exists(ruta_limpia):
            try:
                img = Image.open(ruta_limpia).convert('RGB')
                img_t = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(img_t)
                    prob = torch.nn.functional.softmax(output, dim=1)
                    conf, pred = torch.max(prob, 1)

                categorias = ['APROVECHABLE ✅', 'NO APROVECHABLE ❌']
                # Código de color ANSI para la consola
                color = "\033[92m" if pred.item() == 0 else "\033[91m"
                reset = "\033[0m"

                print(f"\n🔍 ANÁLISIS COMPLETADO:")
                print(f"--------------------------------------------------")
                print(f"RESULTADO: {color}{categorias[pred.item()]}{reset}")
                print(f"CONFIANZA: {conf.item()*100:.2f}%")
                print(f"--------------------------------------------------")
                input("\nPresiona ENTER para probar otra foto...")

            except Exception as e:
                print(f"\n❌ Error procesando la imagen: {e}")
                input("\nPresiona ENTER para reintentar...")
        else:
            print(f"\n⚠️ Archivo no encontrado. Revisa la ruta.")
            input("\nPresiona ENTER para reintentar...")

if __name__ == '__main__':
    ejecutar_interfaz()
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Definimos la clase aquí mismo para NO importar el main.py y evitar el re-entrenamiento
class GarbageDatasetBinario(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        # Mapeo idéntico al entrenamiento: 5 (trash) es No Aprovechable (1)
        target_binario = 1 if target == 5 else 0
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target_binario

def generar_reporte():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, 'Garbage classification')
    model_path = os.path.join(BASE_DIR, 'garbage_model_binario_local.pth')

    print(f"🚀 Generando reporte en: {device}")

    # 2. Cargar Modelo con la ARQUITECTURA CORRECTA (Sequential)
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
    else:
        print("❌ No se encontró el archivo .pth")
        return

    # 3. Preparar Datos
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Cargamos el dataset completo para evaluar el rendimiento final
    dataset_eval = GarbageDatasetBinario(data_dir, transform=transform)
    val_loader = DataLoader(dataset_eval, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []

    print("📊 Evaluando imágenes para la matriz...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if (i+1) % 10 == 0:
                print(f"Lote {i+1} procesado...")

    # 4. Graficar Matriz de Confusión
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Aprovechable', 'No Aprovechable'],
                yticklabels=['Aprovechable', 'No Aprovechable'])
    plt.xlabel('Predicción del Modelo')
    plt.ylabel('Valor Real (Ground Truth)')
    plt.title('Matriz de Confusión Final - Proyecto de Grado UDI')
    
    # Guardar la imagen para tu documento de tesis
    plt.savefig('matriz_confusion_final.png')
    plt.show()

    # 5. Reporte de Métricas (Precision, Recall, F1-Score)
    print("\n📝 REPORTE DETALLADO PARA SUSTENTACIÓN:")
    print("-" * 50)
    print(classification_report(all_labels, all_preds, 
                                target_names=['Aprovechable', 'No Aprovechable']))
    print("-" * 50)

if __name__ == '__main__':
    generar_reporte()
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
# Importamos tu clase del dataset (asegúrate de que esté accesible o cópiala aquí)
from main import GarbageBinaryDataset 

def generar_reporte():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, 'Garbage classification')
    model_path = os.path.join(BASE_DIR, 'garbage_model_binario_local.pth')

    # 1. Cargar Modelo
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 2. Cargar Dataset de Validación
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = GarbageBinaryDataset(data_dir, transform=transform)
    # Usamos un split pequeño o todo el dataset para la matriz final
    val_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []

    print("Evaluando modelo para generar matriz...")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 3. Graficar Matriz
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Aprovechable', 'No Aprovechable'],
                yticklabels=['Aprovechable', 'No Aprovechable'])
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión - Clasificación de Residuos')
    plt.show()

    print("\nReporte Detallado:")
    print(classification_report(all_labels, all_preds, target_names=['Aprovechable', 'No Aprovechable']))

if __name__ == '__main__':
    generar_reporte()
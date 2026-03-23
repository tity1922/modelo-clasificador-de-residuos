import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler

# -------------------- CONFIGURACIÓN --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, 'Garbage classification')
save_path = os.path.join(BASE_DIR, 'garbage_model_binario_local.pth')

batch_size = 32
num_epochs = 10
lr = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------- LÓGICA DE MAPEO (EL TRUCO) --------------------
class GarbageDatasetBinario(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        # Mapeo: 0,1,2,3,4 (glass, paper, etc) -> 0 (Aprovechable)
        #        5 (trash) -> 1 (No Aprovechable)
        target_binario = 1 if target == 5 else 0
        
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target_binario

# -------------------- CARGA DE DATOS --------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Usamos nuestra clase personalizada que mapea las 6 carpetas a 2 etiquetas
full_dataset = GarbageDatasetBinario(data_dir, transform=train_transform)

# Calculamos pesos para el Sampler (Equilibrar los 2300 vs 130)
# Obtenemos las nuevas etiquetas binarias para todos
targets_binarios = torch.tensor([1 if t == 5 else 0 for _, t in full_dataset.samples])
class_sample_count = torch.tensor([(targets_binarios == t).sum() for t in range(2)])
weights = 1. / class_sample_count.float()
samples_weights = weights[targets_binarios]

sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)
train_dl = DataLoader(full_dataset, batch_size=batch_size, sampler=sampler)

# -------------------- MODELO --------------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2) # SIEMPRE 2 SALIDAS
)
model = model.to(device)

# Castigamos más el error en la clase 1 (No Aprovechable)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 20.0]).to(device))
optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

# -------------------- ENTRENAMIENTO --------------------
print(f"🚀 Entrenando con mapeo binario en {device}...")
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_dl:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    print(f"✅ Época {epoch+1} lista.")

torch.save(model.state_dict(), save_path)
print(f"📦 Modelo BINARIO guardado (aunque el dataset tenga 6 carpetas).")
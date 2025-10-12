# ======================================
# AlexNet Pré-Treinada (Treinamento Clássico, sem Técnicas Modernas)
# ======================================
# Objetivo:
#   Reproduzir o comportamento clássico da AlexNet, utilizando pesos pré-treinados no ImageNet
#   e ajustando apenas a camada totalmente conectada (FC) final para classificação binária.
#
# Tipo de Experimento:
#   Classificação binária (ex.: imagens com fissura vs sem fissura)
#   Treinamento tradicional, sem AMP, acumulação de gradientes ou técnicas avançadas.
#
# Arquitetura:
#   AlexNet padrão do torchvision.models, utilizando pesos pré-treinados (ImageNet_Weights.IMAGENET1K_V1)
#   sem modificações estruturais nas camadas convolucionais.
#
# Configurações principais:
#   - Otimizador: SGD clássico com momentum (0.9) e weight decay (5e-4)
#   - Função de perda: BCEWithLogitsLoss (saída binária)
#   - Batch size fixo (sem acumulação)
#   - Acurácia como métrica principal
#   - Logs e gráficos salvos automaticamente
#
# Observações:
#   Este modelo serve como baseline para comparação com versões modernas otimizadas.
# ======================================

#!pip install tqdm -q

import os, gc, time, datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------
# Configurações
# ----------------------
IMG_SIZE = 227
EPOCHS = 30
LR = 1e-4
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR = "/content/DATASET"
OUT_DIR = "/content/drive/MyDrive/VERSAO_FINAL/1_ALEXNET_PRETREINO_SEM_MELHORIAS"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------
# Cria timestamp, log e gráfico
# ----------------------
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_PATH = os.path.join(OUT_DIR, f"1_log_exec_{timestamp}.txt")
GRAPH_PATH = os.path.join(OUT_DIR, f"1_grafico_treinamento_{timestamp}.png")

def log_write(text):
    print(text)
    with open(LOG_PATH, "a") as f:
        f.write(text + "\n")

log_write("🚀 Iniciando treinamento AlexNet Pré-Treinada (Treinamento Clássico, sem técnicas modernas)\n")

# ----------------------
# Transformações (simples, baseadas em ImageNet)
# ----------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ----------------------
# Dataset
# ----------------------
base_dataset = datasets.ImageFolder(DATASET_DIR, transform=test_transform)
classes = base_dataset.classes

total_size = len(base_dataset)
train_size = int(0.7 * total_size)
val_size   = int(0.15 * total_size)
test_size  = total_size - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    base_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_ds.dataset.transform = train_transform
val_ds.dataset.transform   = test_transform
test_ds.dataset.transform  = test_transform

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

log_write(f"Classes: {classes}")
log_write(f"Treino: {len(train_ds)} | Validação: {len(val_ds)} | Teste: {len(test_ds)}\n")

# ----------------------
# Modelo AlexNet (Pré-treinada, sem técnicas modernas)
# ----------------------
from torchvision.models import AlexNet_Weights

model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

# ----------------------
# Função de treino/validação (sem AMP, sem accumulação)
# ----------------------
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="Treinando" if is_train else "Validando", leave=False):
        imgs = imgs.to(DEVICE)
        labels = labels.float().to(DEVICE).unsqueeze(1)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        preds = (torch.sigmoid(outputs) >= 0.5).int()
        correct += (preds == labels.int()).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * imgs.size(0)

    avg_loss = total_loss / len(loader.dataset)
    acc = correct / total
    return avg_loss, acc

# ----------------------
# Treinamento
# ----------------------
MODEL_PATH_FINAL = os.path.join(OUT_DIR, f"1_alexnet_pretreino_final_{timestamp}.pt")
BEST_PATH        = os.path.join(OUT_DIR, f"1_alexnet_pretreino_best_acc_{timestamp}.pt")

history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
best_acc = -1.0

for epoch in range(EPOCHS):
    t0 = time.time()
    train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, criterion)

    history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc);   history["val_acc"].append(val_acc)

    log_write(f"\nEpoch {epoch+1}/{EPOCHS} | {time.time()-t0:.1f}s")
    log_write(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
    log_write(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_acc": best_acc,
            "classes": classes
        }
        torch.save(checkpoint, BEST_PATH)
        log_write(f"  🔥 Novo melhor modelo salvo (acc={best_acc:.4f}) em {BEST_PATH}")

    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

# ----------------------
# Avaliação final
# ----------------------
best_ckpt = torch.load(BEST_PATH, map_location=DEVICE)
model.load_state_dict(best_ckpt["model_state"])
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = (torch.sigmoid(outputs).cpu().numpy() >= 0.5).astype(int)
        y_pred.extend(preds.flatten())
        y_true.extend(labels.numpy())

log_write("\n📋 Relatório de Classificação (melhor ACC):")
log_write(classification_report(y_true, y_pred, target_names=classes))
log_write("Matriz de confusão:")
log_write(str(confusion_matrix(y_true, y_pred)))

# ----------------------
# Gráficos simples
# ----------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.title("Evolução da Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.title("Acurácia")
plt.legend()

plt.tight_layout()
plt.savefig(GRAPH_PATH, dpi=300)
plt.show()

torch.save(model.state_dict(), MODEL_PATH_FINAL)
log_write(f"\n✅ Modelo final salvo em {MODEL_PATH_FINAL}")
log_write(f"🔥 Melhor checkpoint salvo em {BEST_PATH} | ACC={best_ckpt['best_acc']:.4f}")
log_write(f"📊 Gráfico salvo em {GRAPH_PATH}")

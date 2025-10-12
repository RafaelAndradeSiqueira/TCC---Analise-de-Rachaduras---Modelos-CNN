# ======================================
# InceptionV4 Pré-Treinada (Com Melhorias Modernas e AMP)
# ======================================
# Objetivo:
#   Aproveitar os pesos pré-treinados da InceptionV4 (ImageNet)
#   aplicando técnicas modernas de eficiência e estabilidade de treinamento.
#
# Melhorias aplicadas:
#   ✅ AMP (Automatic Mixed Precision) → menor uso de VRAM e maior velocidade.
#   ✅ Acumulação de gradientes → simula batches maiores com menos memória.
#   ✅ Pinagem de memória + non_blocking → transferência CPU↔GPU otimizada.
#   ✅ Channels_last → uso mais eficiente da arquitetura RTX (Tensor Cores).
#   ✅ AdamW + CosineAnnealingLR → otimização estável e convergência suave.
#   ✅ Dropout + BatchNorm → já integrados na arquitetura InceptionV4.
#
# Sem:
#   🚫 Congelamento de camadas
#   🚫 Técnicas que prejudiquem generalização (augmentations agressivos)
#
# ======================================

#!pip install timm tqdm torchmetrics -q

import os, gc, time, datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import timm
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------
# Configurações
# ----------------------
IMG_SIZE = 299
EPOCHS = 30
LR = 1e-4
SUBSET_SIZE = 40000  # opcional (usar subset do dataset)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 16
ACCUM_STEPS = max(2, EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE)
USE_AMP = (DEVICE.type == "cuda")
CHANNELS_LAST = (DEVICE.type == "cuda")

DATASET_DIR = "/content/DATASET"
OUT_DIR = "/content/drive/MyDrive/VERSAO_FINAL/4_INCEPTIONV4_MODERNA"
os.makedirs(OUT_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_PATH = os.path.join(OUT_DIR, f"4_log_exec_{timestamp}.txt")
GRAPH_PATH = os.path.join(OUT_DIR, f"4_grafico_treinamento_{timestamp}.png")
MODEL_PATH_FINAL = os.path.join(OUT_DIR, f"4_inceptionv4_final_{timestamp}.pt")
BEST_PATH = os.path.join(OUT_DIR, f"4_inceptionv4_best_auc_{timestamp}.pt")

def log_write(text):
    print(text)
    with open(LOG_PATH, "a") as f:
        f.write(text + "\n")

# ----------------------
# GPU info
# ----------------------
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    log_write(f"✅ GPU ativa: {gpu_name}")
    log_write(f"💾 Memória total: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB\n")
else:
    log_write("⚠️ Executando em CPU (modo de fallback)\n")

log_write("🚀 Iniciando treinamento InceptionV4 (PRÉ-TREINADA, com técnicas modernas)\n")

# ----------------------
# Transformações moderadas (mantendo compatibilidade com ImageNet)
# ----------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ----------------------
# Dataset e DataLoader
# ----------------------
base_dataset = datasets.ImageFolder(DATASET_DIR, transform=test_transform)
classes = base_dataset.classes

if SUBSET_SIZE and SUBSET_SIZE < len(base_dataset):
    indices = torch.randperm(len(base_dataset))[:SUBSET_SIZE]
    full_dataset = Subset(base_dataset, indices)
else:
    full_dataset = base_dataset

train_size = int(0.7 * len(full_dataset))
val_size   = int(0.15 * len(full_dataset))
test_size  = len(full_dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_ds.dataset.transform = train_transform
val_ds.dataset.transform   = test_transform
test_ds.dataset.transform  = test_transform

train_loader = DataLoader(train_ds, batch_size=MICRO_BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_ds, batch_size=MICRO_BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_ds, batch_size=MICRO_BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True, persistent_workers=True)

log_write(f"Classes: {classes}")
log_write(f"Treino: {len(train_ds)} | Validação: {len(val_ds)} | Teste: {len(test_ds)}")
log_write(f"Micro-batch: {MICRO_BATCH_SIZE} | Accum steps: {ACCUM_STEPS} | Batch efetivo: {MICRO_BATCH_SIZE*ACCUM_STEPS}\n")

# ----------------------
# Modelo InceptionV4 (pré-treinado)
# ----------------------
model = timm.create_model("inception_v4", pretrained=True)
model.last_linear = nn.Linear(model.last_linear.in_features, 1)
model = model.to(DEVICE)

if CHANNELS_LAST:
    model = model.to(memory_format=torch.channels_last)

for param in model.parameters():
    param.requires_grad = True  # sem congelamento

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

metric_acc = BinaryAccuracy().cpu()
metric_auc = BinaryAUROC().cpu()
metric_f1  = BinaryF1Score().cpu()

# ----------------------
# Função de treino/validação com AMP + acumulação
# ----------------------
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    metric_acc.reset(); metric_auc.reset(); metric_f1.reset()

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for step, (imgs, labels) in enumerate(tqdm(loader, desc="Treinando" if is_train else "Validando", leave=False)):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.float().to(DEVICE).unsqueeze(1)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        if is_train:
            scaler.scale(loss / ACCUM_STEPS).backward()
            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            preds = torch.sigmoid(outputs).detach().cpu()
            lbls  = labels.detach().cpu().int()
            metric_acc.update(preds, lbls)
            metric_auc.update(preds, lbls)
            metric_f1.update(preds, lbls)

        total_loss += loss.detach().item() * imgs.size(0)

        del imgs, labels, outputs, loss
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if is_train:
        scheduler.step()

    return (
        total_loss / len(loader.dataset),
        float(metric_acc.compute()),
        float(metric_auc.compute()),
        float(metric_f1.compute())
    )

# ----------------------
# Loop de Treinamento
# ----------------------
history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[],
           "train_auc":[], "val_auc":[], "train_f1":[], "val_f1":[]}

best_auc = -1.0

for epoch in range(EPOCHS):
    t0 = time.time()
    train_loss, train_acc, train_auc, train_f1 = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, val_auc, val_f1 = run_epoch(model, val_loader, criterion)

    history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc);   history["val_acc"].append(val_acc)
    history["train_auc"].append(train_auc);   history["val_auc"].append(val_auc)
    history["train_f1"].append(train_f1);     history["val_f1"].append(val_f1)

    log_write(f"\nEpoch {epoch+1}/{EPOCHS} | {time.time()-t0:.1f}s")
    log_write(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}, auc={train_auc:.4f}, f1={train_f1:.4f}")
    log_write(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}, auc={val_auc:.4f}, f1={val_f1:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_auc": best_auc,
            "classes": classes
        }
        torch.save(checkpoint, BEST_PATH)
        log_write(f"  🔥 Novo melhor AUC ({best_auc:.4f}) salvo em {BEST_PATH}")

# ----------------------
# Avaliação Final
# ----------------------
best_ckpt = torch.load(BEST_PATH, map_location=DEVICE)
model.load_state_dict(best_ckpt["model_state"])
model.eval()

y_true, y_pred = [], []
with torch.no_grad(), torch.cuda.amp.autocast(enabled=USE_AMP):
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        outputs = model(imgs)
        preds = (torch.sigmoid(outputs).cpu().numpy() >= 0.5).astype(int)
        y_pred.extend(preds.flatten())
        y_true.extend(labels.numpy())

log_write("\n📋 Relatório de Classificação (melhor AUC):")
log_write(classification_report(y_true, y_pred, target_names=classes))
log_write("Matriz de confusão:")
log_write(str(confusion_matrix(y_true, y_pred)))

# ----------------------
# Gráficos e Salvamento
# ----------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.title("Evolução da Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history["val_acc"], label="Val Acc")
plt.plot(history["val_auc"], label="Val AUC")
plt.plot(history["val_f1"], label="Val F1")
plt.title("Métricas de Validação")
plt.legend()

plt.tight_layout()
plt.savefig(GRAPH_PATH, dpi=300)
plt.show()

torch.save(model.state_dict(), MODEL_PATH_FINAL)
log_write(f"\n✅ Modelo final salvo em {MODEL_PATH_FINAL}")
log_write(f"🔥 Melhor checkpoint salvo em {BEST_PATH} | AUC={best_ckpt['best_auc']:.4f}")
log_write(f"📊 Gráfico salvo em {GRAPH_PATH}")
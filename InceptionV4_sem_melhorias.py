# ======================================
# InceptionV4 Pré-Treinada (Padrão Original, sem Técnicas Modernas)
# ======================================
!pip install -q timm tqdm

import os, gc, time, datetime, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score

# ----------------------
# Configurações básicas
# ----------------------
IMG_SIZE   = 299
EPOCHS     = 30
LR         = 1e-4
BATCH_SIZE = 32                 
SEED       = 42


DATASET_DIR = "/content/DATASET/DATASET"   
OUT_DIR     = "/content/drive/MyDrive/VERSAO_FINAL/2-VERSAO/2.2-INCEPTIONV4/2.2.2-FINI-TUNING-COMPLETO_ORIGINAL/A100"
os.makedirs(OUT_DIR, exist_ok=True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# ----------------------
# Timestamp e logs
# ----------------------
timestamp  = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_PATH   = os.path.join(OUT_DIR, f"3_log_exec_{timestamp}.txt")
GRAPH_PATH = os.path.join(OUT_DIR, f"3_grafico_treinamento_{timestamp}.png")

def log_write(text):
    print(text)
    with open(LOG_PATH, "a") as f:
        f.write(text + "\n")

log_write(" Iniciando treinamento InceptionV4 (PRÉ-TREINADA, RMSProp, best=ACC, sem AMP/accum)\n")

# ----------------------
# Transforms (padrão ImageNet)
# ----------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ----------------------
# Dataset & Loaders 
# ----------------------
def build_splits(root, seed=42):
    root = os.path.abspath(root)
    has_split = all(os.path.isdir(os.path.join(root, d)) for d in ["train","val","test"])
    if has_split:
        train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=train_transform)
        val_ds   = datasets.ImageFolder(os.path.join(root, "val"),   transform=test_transform)
        test_ds  = datasets.ImageFolder(os.path.join(root, "test"),  transform=test_transform)
        classes  = train_ds.classes
    else:
        base_ds = datasets.ImageFolder(root, transform=test_transform)
        classes = base_ds.classes
        total   = len(base_ds)
        train_n = int(0.7 * total)
        val_n   = int(0.15 * total)
        test_n  = total - train_n - val_n
        gen = torch.Generator().manual_seed(seed)
        train_ds, val_ds, test_ds = random_split(base_ds, [train_n, val_n, test_n], generator=gen)
        train_ds.dataset.transform = train_transform
        val_ds.dataset.transform   = test_transform
        test_ds.dataset.transform  = test_transform
    return train_ds, val_ds, test_ds, classes

train_ds, val_ds, test_ds, classes = build_splits(DATASET_DIR, SEED)

num_workers = 4
pin_mem     = (DEVICE.type == "cuda")
persist     = num_workers > 0

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_mem,
                          persistent_workers=persist)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_mem,
                          persistent_workers=persist)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_mem,
                          persistent_workers=persist)

log_write(f"Classes: {classes}")
log_write(f"Treino: {len(train_ds)} | Validação: {len(val_ds)} | Teste: {len(test_ds)}\n")

# ----------------------
# Modelo InceptionV4 (pré-treinada no ImageNet)
# ----------------------
model = timm.create_model("inception_v4", pretrained=True, num_classes=1)  # saída (B,1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

# Otimizador clássico do paper
optimizer = optim.RMSprop(
    model.parameters(),
    lr=LR,
    alpha=0.9,
    eps=1.0,
    momentum=0.9,
    weight_decay=5e-4
)

# ----------------------
# Função de treino/validação 
# ----------------------
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="Treinando" if is_train else "Validando", leave=False):
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.float().to(DEVICE, non_blocking=True).unsqueeze(1)

        outputs = model(imgs)
        loss    = criterion(outputs, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        preds   = (torch.sigmoid(outputs) >= 0.5).int()
        correct += (preds == labels.int()).sum().item()
        total   += labels.size(0)
        total_loss += loss.item() * imgs.size(0)

    avg_loss = total_loss / len(loader.dataset)
    acc      = correct / total
    return avg_loss, acc

# ----------------------
# Treinamento (best por ACC, como no original)
# ----------------------
MODEL_PATH_FINAL = os.path.join(OUT_DIR, f"3_inceptionv4_final_{timestamp}.pt")
BEST_PATH        = os.path.join(OUT_DIR, f"3_inceptionv4_best_acc_{timestamp}.pt")

history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
best_acc = -1.0

for epoch in range(EPOCHS):
    t0 = time.time()
    train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
    val_loss,   val_acc   = run_epoch(model, val_loader,   criterion)

    history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc);   history["val_acc"].append(val_acc)

    elapsed = time.time() - t0
    log_write(f"\nEpoch {epoch+1}/{EPOCHS} | Tempo: {elapsed:.1f}s")
    log_write(f"  Train → loss={train_loss:.4f}, acc={train_acc:.4f}")
    log_write(f"  Val   → loss={val_loss:.4f}, acc={val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_acc": best_acc,
            "classes": classes,
        }
        torch.save(checkpoint, BEST_PATH)
        log_write(f"   Novo melhor modelo salvo (acc={best_acc:.4f}) em {BEST_PATH}")

    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

# ----------------------
# Avaliação final 
# ----------------------
best_ckpt = torch.load(BEST_PATH, map_location=DEVICE)
model.load_state_dict(best_ckpt["model_state"])
model.eval()

y_true, y_prob, y_pred = [], [], []
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Testando", leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        logits = model(imgs)                    
        probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        preds  = (probs >= 0.5).astype(int)

        y_prob.extend(probs.tolist())
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().astype(int).tolist())

acc_test = accuracy_score(y_true, y_pred)
f1_test  = f1_score(y_true, y_pred)
try:
    auc_test = roc_auc_score(y_true, y_prob)
except ValueError:
    auc_test = float("nan")

log_write("\n Relatório de Classificação (melhor ACC no TESTE):")
log_write(classification_report(y_true, y_pred, target_names=classes))
log_write("Matriz de confusão:")
log_write(str(confusion_matrix(y_true, y_pred)))
log_write(f"\n Métricas no TESTE → Acc={acc_test:.6f} | F1={f1_test:.6f} | AUC={auc_test:.6f}")

# ----------------------
# Gráficos simples
# ----------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"],   label="Val Loss")
plt.title("Evolução da Loss"); plt.legend()

plt.subplot(1,2,2)
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"],   label="Val Acc")
plt.title("Acurácia"); plt.legend()

plt.tight_layout()
plt.savefig(GRAPH_PATH, dpi=300)
plt.show()

torch.save(model.state_dict(), MODEL_PATH_FINAL)
log_write(f"\n Modelo final salvo em {MODEL_PATH_FINAL}")
log_write(f" Melhor checkpoint salvo em {BEST_PATH} | ACC={best_ckpt['best_acc']:.4f}")
log_write(f" Gráfico salvo em {GRAPH_PATH}")

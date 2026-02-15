import os
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
# SETTINGS
# -----------------------------
DATA_DIR = "data"          # должно быть: data/train, data/val, data/test
BATCH_SIZE = 32
EPOCHS = 5                 # поставил 5, чтобы быстрее; можно увеличить
LR = 1e-3
SEED = 42
LOG_EVERY_N_BATCHES = 20   # вывод прогресса каждые N батчей
NUM_WORKERS = 0            # для Windows чаще стабильнее 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)

print("STEP 1: script started", flush=True)
print(f"Device: {DEVICE}", flush=True)

# -----------------------------
# TRANSFORMS (augmentation only for train)
# -----------------------------
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -----------------------------
# ROBUST DATASET (skip broken images)
# -----------------------------
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            # принудительно проверяем чтение
            with Image.open(path) as img:
                img = img.convert("RGB")
            # стандартный loader torchvision тоже ок, но тут уже проверили
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target
        except Exception as e:
            # возвращаем None, чтобы коллатор выкинул этот пример
            return None

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(ys)

# -----------------------------
# LOAD DATASETS
# -----------------------------
train_path = os.path.join(DATA_DIR, "train")
val_path   = os.path.join(DATA_DIR, "val")
test_path  = os.path.join(DATA_DIR, "test")

for p in [train_path, val_path, test_path]:
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Не найдена папка: {p}. Проверь, что есть data/train, data/val, data/test")

print("STEP 2: loading datasets...", flush=True)

train_ds = SafeImageFolder(train_path, transform=train_tfms)
val_ds   = SafeImageFolder(val_path, transform=eval_tfms)
test_ds  = SafeImageFolder(test_path, transform=eval_tfms)

class_names = train_ds.classes
num_classes = len(class_names)

print(f"Classes: {num_classes}", flush=True)
print("Train images:", len(train_ds), "Val images:", len(val_ds), "Test images:", len(test_ds), flush=True)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, collate_fn=safe_collate
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, collate_fn=safe_collate
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, collate_fn=safe_collate
)

# -----------------------------
# MODEL: ResNet18 fine-tuning
# -----------------------------
print("STEP 3: loading model (ResNet18 pretrained)...", flush=True)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# TRAIN / EVAL
# -----------------------------
def run_epoch(loader, train=False, epoch=0):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    start = time.time()
    batch_idx = 0

    for batch in loader:
        if batch is None:
            continue

        x, y = batch
        x, y = x.to(DEVICE), y.to(DEVICE)

        if train:
            optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_seen += bs

        batch_idx += 1
        if batch_idx % LOG_EVERY_N_BATCHES == 0:
            elapsed = time.time() - start
            avg_loss = total_loss / max(total_seen, 1)
            avg_acc = total_correct / max(total_seen, 1)
            mode = "train" if train else "val"
            print(f"  [{mode}] epoch {epoch} batch {batch_idx} | seen {total_seen} | loss {avg_loss:.4f} acc {avg_acc:.4f} | {elapsed:.1f}s",
                  flush=True)

    avg_loss = total_loss / max(total_seen, 1)
    avg_acc = total_correct / max(total_seen, 1)
    return avg_loss, avg_acc

best_val_acc = 0.0
best_path = "best_model.pth"

print("STEP 4: starting training loop...", flush=True)

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True, epoch=epoch)
    va_loss, va_acc = run_epoch(val_loader, train=False, epoch=epoch)

    print(f"Epoch {epoch}/{EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}",
          flush=True)

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        torch.save(model.state_dict(), best_path)
        print(f"  ✅ Saved new best model: {best_path} (val acc {best_val_acc:.4f})", flush=True)

print(f"\nBest val acc: {best_val_acc:.4f}", flush=True)
print("Loading best model for test...", flush=True)

# -----------------------------
# TEST + METRICS
# -----------------------------
model.load_state_dict(torch.load(best_path, map_location=DEVICE))
model.eval()

all_preds = []
all_true = []

with torch.no_grad():
    for batch in test_loader:
        if batch is None:
            continue
        x, y = batch
        x = x.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_true.append(y.numpy())

all_preds = np.concatenate(all_preds) if len(all_preds) else np.array([], dtype=int)
all_true = np.concatenate(all_true) if len(all_true) else np.array([], dtype=int)

if len(all_true) == 0:
    raise RuntimeError("На test не прочиталось ни одного изображения. Проверь data/test и файлы.")

cm = confusion_matrix(all_true, all_preds, labels=list(range(num_classes)))

# per-class accuracy
print("\nPer-class accuracy:", flush=True)
per_class_acc = {}
for i, name in enumerate(class_names):
    row_sum = cm[i].sum()
    acc_i = (cm[i, i] / row_sum) if row_sum > 0 else 0.0
    per_class_acc[name] = acc_i
    print(f"{name:20s} : {acc_i:.4f}   (support={row_sum})", flush=True)

test_acc = (all_preds == all_true).mean()
print(f"\nTest accuracy: {test_acc:.4f}", flush=True)

# confusion matrix plot
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()
ticks = np.arange(num_classes)
plt.xticks(ticks, class_names, rotation=90)
plt.yticks(ticks, class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
print("\nSaved confusion matrix to: confusion_matrix.png", flush=True)
print("DONE ✅", flush=True)

import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# -----------------------------
# SETTINGS
# -----------------------------
DATA_DIR = "data"
MODEL_PATH = "best_model.pth"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model and test data...", flush=True)

# -----------------------------
# TRANSFORMS (no augmentation)
# -----------------------------
eval_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -----------------------------
# LOAD TEST DATA
# -----------------------------
test_path = os.path.join(DATA_DIR, "test")

if not os.path.isdir(test_path):
    raise FileNotFoundError("Папка data/test не найдена")

test_ds = datasets.ImageFolder(test_path, transform=eval_tfms)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

class_names = test_ds.classes
num_classes = len(class_names)

print(f"Classes: {num_classes}")
print(f"Test images: {len(test_ds)}")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print("Model loaded. Starting evaluation...\n")

# -----------------------------
# EVALUATION
# -----------------------------
all_preds = []
all_true = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_true.append(y.numpy())

all_preds = np.concatenate(all_preds)
all_true = np.concatenate(all_true)

# -----------------------------
# METRICS
# -----------------------------
cm = confusion_matrix(all_true, all_preds, labels=list(range(num_classes)))

print("Per-class accuracy:")
per_class_acc = {}

for i, name in enumerate(class_names):
    row_sum = cm[i].sum()
    acc_i = (cm[i, i] / row_sum) if row_sum > 0 else 0.0
    per_class_acc[name] = acc_i
    print(f"{name:20s} : {acc_i:.4f}   (support={row_sum})")

test_acc = (all_preds == all_true).mean()
print(f"\nOverall Test accuracy: {test_acc:.4f}")

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
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

print("\nConfusion matrix saved as confusion_matrix.png")
print("DONE")

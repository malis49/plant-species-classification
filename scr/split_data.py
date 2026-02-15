import os
import random
import shutil

# Путь к исходным данным (где лежат 12 папок классов)
source_dir = "dataset"   # <-- если папка называется иначе, поменяй

# Куда будем раскладывать
base_dir = "data"

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Создаем папки
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(base_dir, split), exist_ok=True)

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    train_split = int(len(images) * train_ratio)
    val_split = int(len(images) * (train_ratio + val_ratio))

    splits = {
        "train": images[:train_split],
        "val": images[train_split:val_split],
        "test": images[val_split:]
    }

    for split_name, split_images in splits.items():
        split_class_dir = os.path.join(base_dir, split_name, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for img in split_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_class_dir, img)
            shutil.copy(src, dst)

print("Готово! Данные разделены.")

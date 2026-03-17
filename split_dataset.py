"""
Script chia dataset từ data/original/ thành dataset/train, dataset/val, dataset/test
Tỉ lệ: 70% train / 15% val / 15% test
"""
import os
import shutil
import random

SRC = "data/original/"
DST = "dataset/"
SPLIT = (0.70, 0.15, 0.15)

random.seed(42)

for cls in sorted(os.listdir(SRC)):
    cls_path = os.path.join(SRC, cls)
    if not os.path.isdir(cls_path):
        continue

    imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
    random.shuffle(imgs)
    n = len(imgs)
    cut1 = int(n * SPLIT[0])
    cut2 = int(n * (SPLIT[0] + SPLIT[1]))

    splits = {
        'train': imgs[:cut1],
        'val':   imgs[cut1:cut2],
        'test':  imgs[cut2:]
    }

    for split_name, file_list in splits.items():
        dst_dir = os.path.join(DST, split_name, cls)
        os.makedirs(dst_dir, exist_ok=True)
        for img in file_list:
            shutil.copy2(os.path.join(cls_path, img), os.path.join(dst_dir, img))

    print(f"  {cls:12s} -> train: {len(splits['train']):4d} | val: {len(splits['val']):4d} | test: {len(splits['test']):4d}")

print("\nDataset split done!")

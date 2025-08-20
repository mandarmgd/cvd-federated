import os
import shutil
from pathlib import Path
from math import ceil

# Original folders
base_dir = Path("ECG Dataset")  
class_folders = ["Abnormal heartbeat", "History of MI", "Normal Person"]

# Create training set folders
output_base = base_dir / "TrainingSets"
output_base.mkdir(exist_ok=True)

training_sets = [output_base / f"TrainingSet{i}" for i in range(1, 4)]
for tset in training_sets:
    for cls in class_folders:
        (tset / cls).mkdir(parents=True, exist_ok=True)

# Distribute images
for cls in class_folders:
    cls_path = base_dir / cls
    images = sorted([img for img in cls_path.iterdir() if img.is_file()])
    split_size = ceil(len(images) / 3)

    for i in range(3):
        subset = images[i*split_size:(i+1)*split_size]
        for img in subset:
            shutil.copy(img, training_sets[i] / cls)

print("Images distributed successfully into three training sets.")
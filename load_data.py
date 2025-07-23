import os
import numpy as np
import cv2


def load_dataset(data_dir="data", image_size=(64, 64)):
    X = []
    y = []
    class_names = sorted(os.listdir(data_dir))

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for file in os.listdir(class_path):
            file_path = os.path.join(class_path, file)
            try:
                img = cv2.imread(file_path)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    X.append(img)
                    y.append(label)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    print(f"✅ Loaded images: {np.array(X).shape}")
    print(f"✅ Labels: {np.array(y).shape}")
    return np.array(X), np.array(y)

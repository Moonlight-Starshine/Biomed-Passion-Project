from importlib.resources import path
import os
import cv2

import numpy as np
import pandas as pd

print("SCRIPT STARTED")


def extract_features(img):
    img = cv2.resize(img, (1224, 960))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_channel = img_lab[:, :, 1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=5
    )
   
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)
   

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rbc_areas = []
    rbc_red_values = []
    rbc_red_std_values = []
    pale_ratios = []
    rbc_a_values = []

    for c in contours:
        area = cv2.contourArea(c)

        if 50 < area < 5000:
            rbc_areas.append(area)

            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)

            red_channel = img_rgb[:, :, 0]
            pixels = red_channel[mask == 255]
            a_pixels = a_channel[mask == 255]
            if len(a_pixels) > 0:
                rbc_a_values.append(np.mean(a_pixels))

            if len(pixels) > 0:
                # color features
                rbc_red_values.append(np.mean(pixels))
                rbc_red_std_values.append(np.std(pixels))

                # pale ratio
                threshold = np.percentile(pixels, 25)
                pale_pixels = np.sum(pixels < threshold)
                total_pixels = len(pixels)

                if total_pixels > 0:
                    pale_ratio = pale_pixels / total_pixels
                    pale_ratios.append(pale_ratio)
                    


    print(f"Detected {len(rbc_areas)} cells")

    if len(rbc_areas) == 0:
        return None

    # aggregate AFTER loop
    mean_area = np.mean(rbc_areas)
    std_area = np.std(rbc_areas)
    mean_red = np.mean(rbc_red_values) if rbc_red_values else 0
    rbc_count = len(rbc_areas)

    mean_red_std = np.mean(rbc_red_std_values) if rbc_red_std_values else 0
    mean_pale_ratio = np.mean(pale_ratios) if pale_ratios else 0
    mean_a = np.mean(rbc_a_values) if rbc_a_values else 0
    
    return [
        mean_area,
        std_area,
        mean_red,
        rbc_count,
        mean_red_std,
        mean_pale_ratio,
        mean_a
    ]

if __name__ == "__main__":
# -------------------------------
# BUILD DATASET
# -------------------------------
<<<<<<< HEAD
data = []
base_path = "dataset"

for label_name in ["anemia", "normal"]:
    folder = os.path.join(base_path, label_name)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        img = cv2.imread(path)
        if img is None:
            continue

        features = extract_features(img)
        if features is None:
            continue

        label = 1 if label_name == "anemia" else 0
        data.append(features + [label])


# -------------------------------
# SAVE CSV
# -------------------------------
df = pd.DataFrame(data, columns=[
    "mean_area",
    "std_area",
    "mean_red",
    "rbc_count",
    "red_std",
    "pale_ratio",
    "mean_a",
    "label"
])

df.to_csv("real_dataset.csv", index=False)

print("✅ Dataset created: real_dataset.csv")
print(df.head())
=======
    data = []
    base_path = "dataset"
    
    for label_name in ["anemia", "normal"]:
        folder = os.path.join(base_path, label_name)
    
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
    
            img = cv2.imread(path)
            if img is None:
                continue
    
            features = extract_features(img)
            if features is None:
                continue
    
            label = 1 if label_name == "anemia" else 0
            data.append(features + [label])
    
    
    # -------------------------------
    # SAVE CSV
    # -------------------------------
    df = pd.DataFrame(data, columns=[
        "mean_area",
        "std_area",
        "mean_red",
        "rbc_count",
        "red_std",
        "pale_ratio",
        "label"
    ])
    
    df.to_csv("real_dataset.csv", index=False)
    
    print("✅ Dataset created: real_dataset.csv")
    print(df.head())
>>>>>>> 4b1ca77064ca6dabad0ee068ff3b0899f90ae381

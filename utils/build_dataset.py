import os
import cv2
import numpy as np
import pandas as pd

print("SCRIPT STARTED")


def extract_features(img):
    img = cv2.resize(img, (256, 256))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rbc_areas = []
    rbc_red_values = []
    rbc_red_std_values = []
    pale_ratios = []

    for c in contours:
        area = cv2.contourArea(c)

        if 50 < area < 5000:
            rbc_areas.append(area)

            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)

            red_channel = img_rgb[:, :, 0]
            pixels = red_channel[mask == 255]

            if len(pixels) > 0:
                # color features
                rbc_red_values.append(np.mean(pixels))
                rbc_red_std_values.append(np.std(pixels))

                # pale ratio
                pale_pixels = np.sum(pixels < 100)
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

    return [
        mean_area,
        std_area,
        mean_red,
        rbc_count,
        mean_red_std,
        mean_pale_ratio
    ]


# -------------------------------
# BUILD DATASET
# -------------------------------
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
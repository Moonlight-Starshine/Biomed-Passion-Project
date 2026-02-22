# Disease Detection from Microscopic Slides — Setup Guide

## Quick Start (2 min)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Organize your images
#    Create: data/raw/{disease1}/{image1.jpg}, data/raw/{disease2}/{image2.jpg}, ...
#    Example:
#      data/raw/malaria/cell_1.jpg
#      data/raw/malaria/cell_2.jpg
#      data/raw/healthy/cell_3.jpg
#      data/raw/healthy/cell_4.jpg

# 3. Train the model
python src/train.py --data_dir data/raw --num_epochs 50 --architecture resnet50

# 4. (Optional) Predict on new images
python src/predict.py path/to/image.jpg --model models/resnet50_best.pth
```

---

## 1. Data Structure

Your images must be organized as:
```
data/raw/
├── disease_class_1/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
├── disease_class_2/
│   ├── image_1.jpg
│   └── ...
└── disease_class_n/
    └── ...
```

**Examples:**
- `data/raw/blood_cancer/slide_001.jpg`
- `data/raw/normal_blood/slide_001.jpg`
- `data/raw/mycobacterium/slide_001.jpg`

The script will automatically detect class names from folder names.

**Minimum images per class:** 10-20 (more is better; 100+ recommended for production).

---

## 2. Installation

### Option A: conda (recommended)
```bash
conda create -n disease-detection python=3.10
conda activate disease-detection
pip install -r requirements.txt
```

### Option B: pip (with virtualenv)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### GPU Support (CUDA)
If you have NVIDIA GPU:
```bash
# Uninstall CPU torch
pip uninstall torch torchvision

# Install GPU version (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 3. Training

### Basic Training
```bash
python src/train.py \
  --data_dir data/raw \
  --num_epochs 50 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --architecture resnet50
```

### Advanced Options
```bash
python src/train.py \
  --data_dir data/raw \
  --output_dir models_v2 \
  --num_epochs 100 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --architecture mobilenet_v2 \
  --device cuda \
  --seed 42
```

**Available architectures:**
- `resnet50`: Larger, more accurate (~350MB). Better for research/cloud.
- `mobilenet_v2`: Smaller, faster (~14MB). Better for edge/mobile deployment.

**Expected outcomes:**
- Training takes 5–30 min (depends on image count and GPU)
- Best model saved to `models/resnet50_best.pth`
- Training history saved to `models/training_history.json`
- Config saved to `models/config.json`

---

## 4. Prediction (Inference)

### Single Image
```bash
python src/predict.py /path/to/image.jpg \
  --model models/resnet50_best.pth \
  --config models/config.json
```

Output:
```
Using device: cpu
Predicted: malaria
Confidence: 92.34%

All probabilities:
  malaria: 0.9234
  healthy: 0.0766
```

### Batch Processing (Directory)
```bash
python src/predict.py test_images/ \
  --model models/resnet50_best.pth \
  --config models/config.json \
  --batch
```

---

## 5. Evaluation & Visualization

### Load and evaluate in Python/Jupyter:
```python
import json
import torch
from src.train import evaluate_model
from src.model import get_model
from src.data import get_dataloaders
from src.evaluate import (
    get_detailed_metrics, plot_confusion_matrix,
    plot_training_history, plot_roc_curves
)

# Load config
with open('models/config.json') as f:
    config = json.load(f)

# Load dataloaders
_, val_loader, test_loader, disease_names = get_dataloaders(
    data_dir='data/raw', batch_size=32
)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(config['num_classes'], config['architecture'])
model.load_state_dict(torch.load('models/resnet50_best.pth'))
model.to(device)

# Get metrics
cm, preds, labels = get_detailed_metrics(model, test_loader, device, disease_names)
plot_confusion_matrix(cm, disease_names, save_path='results/confusion_matrix.png')

# Plot history
with open('models/training_history.json') as f:
    history = json.load(f)
plot_training_history(history, save_path='results/training_curves.png')

# Plot ROC
plot_roc_curves(model, test_loader, device, disease_names, 
                config['num_classes'], save_path='results/roc_curves.png')
```

---

## 6. Web App (Streamlit)

Run an interactive web interface:
```bash
streamlit run src/app.py
```

Then navigate to `http://localhost:8501`.

---

## 7. Troubleshooting

### "No images found in data/raw"
- Ensure folders are named with disease classes.
- Check that images are in subdirectories, not the root `data/raw/` folder.
- Verify image file extensions are `.jpg`, `.jpeg`, or `.png`.

### CUDA out of memory
- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Use mobilenet_v2: `--architecture mobilenet_v2`
- Use CPU: `--device cpu`

### Low accuracy
- **More data:** Train on 100+ images per class.
- **More epochs:** Try `--num_epochs 100` or `--num_epochs 200`.
- **Different architecture:** Try `--architecture mobilenet_v2`.
- **Longer training:** Reduce learning rate: `--learning_rate 0.0001`.
- **Data augmentation:** Already enabled by default.

### Slow training
- Reduce image size in `data.py`: `image_size = 128` (default 224).
- Reduce batch size to fit more in GPU memory.
- Use GPU (install CUDA support).

---

## 8. Project Structure

```
.
├── data/
│   ├── raw/               # Your images (organized by disease)
│   └── processed/         # Cached preprocessed data
├── models/                # Trained models & checkpoints
├── src/
│   ├── app.py            # Streamlit web app
│   ├── data.py           # Data loading & preprocessing
│   ├── model.py          # Model architectures
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation & metrics
│   └── predict.py        # Inference script
├── notebooks/            # Jupyter notebooks
├── requirements.txt      # Python dependencies
├── CONFIG.yaml          # Configuration
├── SETUP.md             # This file
└── README.md            # Project overview
```

---

## 9. Next Steps

1. **Collect/download data:** Use public datasets (Camelyon, BreakHis, TCGA, etc.).
2. **Annotate images:** Label images with disease class.
3. **Organize in `data/raw/`** and train.
4. **Validate results** with confusion matrix and per-class metrics.
5. **Deploy:** Export model for production use.

---

## 10. Resources

- **PyTorch:** https://pytorch.org/
- **Histopathology Datasets:**
  - Camelyon16/17: https://camelyon16.grand-challenge.org/
  - BreakHis: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
  - TCGA: https://portal.gdc.cancer.gov/
  - MedMNIST: https://medmnist.github.io/

---

**Happy training! 🔬 If you have issues, check the troubleshooting section above.**

# Biomed-Passion-Project

## 📌 NEW: Complete Disease Detection Pipeline

**👉 [START HERE](START_HERE.md)** for the complete disease classification setup!

This includes:
- ✅ Data loading & preprocessing (stain normalization, augmentation)
- ✅ Model training (ResNet50 & MobileNetV2)
- ✅ Full evaluation pipeline (confusion matrices, ROC curves, per-class metrics)
- ✅ Inference on new images
- ✅ Streamlit web app for easy predictions
- ✅ Jupyter notebook for interactive analysis

**Quick start:**
```bash
pip install -r requirements.txt
python src/train.py --data_dir data/raw --num_epochs 50
```

See [START_HERE.md](START_HERE.md) for complete instructions!

---

## Description (Original)

Build a low-cost AI tool that attaches to any home microscope (or even a smartphone microscope) and:
- automatically counts cells
- measures cell size
- flags abnormalities
- outputs a simple "analysis report"

### Why this is impactful
Cell counting is essential for:
- anemia screening (RBC counts)
- infection monitoring (WBC counts)
- yeast viability for biotechnology
- water contamination checks

But automated cell counters cost $1,000–$6,000. You could build one for under $20 using AI + a microscope.

### Wet-lab experiment portion
Use:
- yeast suspensions, or
- pre-prepared blood smear slides, or
- pond water microbes

Collect microscope images → annotate ~100–200 manually → train a light CNN or use classical computer vision (OpenCV).

### Deliverables
- A simple web app / Python script that counts cells from an uploaded photo
- A GitHub repository of your code
- A printable "attachable guide" showing how to align a phone with the microscope
- Demo images + accuracy comparisons

## Installation

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run src/app.py`

## Usage

Upload a microscope image, and get the cell count, sizes, and report.

## Guide

See [guide/attachable_guide.md](guide/attachable_guide.md) for printable instructions.
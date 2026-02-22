# 🚀 Complete Setup — Just Copy & Paste

This file contains all commands to get started with disease detection from microscopic slides.

---

## ✅ Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

On macOS with Apple Silicon:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 📸 Step 2: Create Sample Data (Optional — for immediate testing)

```bash
python CREATE_SAMPLE_DATA.py
```

This generates 30 synthetic images (10 per disease class):
- `data/raw/malaria/` (10 images)
- `data/raw/healthy/` (10 images)
- `data/raw/tuberculosis/` (10 images)

**Or use your own data:**
```
data/raw/
├── class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_2/
│   └── ...
```

---

## 🎯 Step 3: Train the Model

### Quick test (5 minutes):
```bash
python src/train.py --data_dir data/raw --num_epochs 5 --batch_size 16
```

### Production (30 minutes):
```bash
python src/train.py \
  --data_dir data/raw \
  --num_epochs 50 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --architecture resnet50
```

### Fast lightweight model (edge deployment):
```bash
python src/train.py \
  --data_dir data/raw \
  --num_epochs 50 \
  --architecture mobilenet_v2 \
  --batch_size 32
```

**Output files:**
- `models/resnet50_best.pth` — Model weights
- `models/config.json` — Configuration
- `models/training_history.json` — Loss & accuracy curves

---

## 🔮 Step 4: Predict on New Images

### Single image:
```bash
python src/predict.py path/to/image.jpg \
  --model models/resnet50_best.pth \
  --config models/config.json
```

### Batch directory:
```bash
python src/predict.py path/to/images/ \
  --model models/resnet50_best.pth \
  --config models/config.json \
  --batch
```

---

## 🌐 Step 5: Web App (Interactive UI)

```bash
streamlit run src/app.py
```

Then open: **http://localhost:8501**

---

## 📊 Step 6: Evaluate & Visualize (Jupyter)

```bash
jupyter notebook notebooks/quickstart.ipynb
```

Or run the notebook steps to:
- Load data
- Train model
- Generate confusion matrix
- Plot training curves
- Make predictions
- View per-class metrics

---

## 🎨 Project Structure

```
.
├── data/
│   ├── raw/                    ← Your images here
│   └── processed/
├── models/
│   ├── resnet50_best.pth       ← Trained model
│   ├── config.json             ← Class names & config
│   └── training_history.json   ← Loss curves
├── src/
│   ├── app.py                  ← Streamlit web UI
│   ├── data.py                 ← Data loading
│   ├── model.py                ← Model architectures
│   ├── train.py                ← Training script
│   ├── evaluate.py             ← Metrics & plots
│   └── predict.py              ← Inference
├── notebooks/
│   └── quickstart.ipynb        ← Interactive demo
├── requirements.txt
├── CONFIG.yaml
├── SETUP.md                    ← Detailed guide
└── README.md                   ← Full documentation
```

---

## ⚙️ Architecture Choices

| Architecture | Model Size | Speed | Accuracy | Best For |
|---|---|---|---|---|
| **ResNet50** | 350 MB | Medium | High | Research, cloud |
| **MobileNetV2** | 14 MB | Fast | Good | Mobile, edge, deployment |

Use `--architecture resnet50` or `--architecture mobilenet_v2` in train.py.

---

## 🔧 Hyperparameter Tuning

```bash
# More training time → Better accuracy
python src/train.py --num_epochs 100

# Smaller batches → Slower but more stable
python src/train.py --batch_size 8

# Smaller learning rate → More careful learning
python src/train.py --learning_rate 0.0001

# Faster training (lower quality)
python src/train.py --num_epochs 20 --batch_size 64
```

---

## 🐛 Common Issues & Fixes

**"No images found in data/raw"**
```bash
# Make sure structure is: data/raw/{disease_name}/*.jpg
ls data/raw/  # Should show disease directories
```

**Out of GPU memory**
```bash
python src/train.py --batch_size 8 --architecture mobilenet_v2 --device cpu
```

**Slow training**
```bash
python src/train.py --num_epochs 20 --batch_size 64 --architecture mobilenet_v2
```

**Low accuracy**
```bash
# Get more data (100+ images per class) and train longer
python src/train.py --num_epochs 200 --learning_rate 0.0001
```

---

## 📚 Next Steps

1. **Collect real data** (or use public datasets listed in SETUP.md)
2. **Organize into** `data/raw/disease_name/*.jpg`
3. **Run training** (use quick test first to verify setup)
4. **Check metrics** in Jupyter notebook
5. **Deploy** via web app, API, or batch inference

---

## 📖 Full Documentation

- **[SETUP.md](SETUP.md)** — Detailed installation & troubleshooting
- **[README.md](README.md)** — Project overview & features
- **[CONFIG.yaml](CONFIG.yaml)** — All configuration options
- **[notebooks/quickstart.ipynb](notebooks/quickstart.ipynb)** — Interactive walkthrough

---

## 🎓 Expected Results

With sample data:
- Train time: 2–5 minutes
- Accuracy: 60–75% (synthetic data is difficult!)
- Model size: 350 MB (ResNet50) or 14 MB (MobileNetV2)

With real histopathology data (100+ images per class):
- Train time: 10–60 minutes
- Accuracy: 85–99% (depends on disease complexity)
- Generalization: High

---

**Ready to start? Run this:**

```bash
# If you have data already:
python src/train.py --data_dir data/raw --num_epochs 50

# If you want to test first:
python CREATE_SAMPLE_DATA.py
python src/train.py --data_dir data/raw --num_epochs 5
```

Then check `models/` for your trained model! 🎉

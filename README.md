
# Intel Image Classifier

A web application that classifies natural scene images into 6 categories using CNN models trained with both **PyTorch** and **TensorFlow**.

**Live demo:** https://huggingface.co/spaces/harilova12/intel-image-classifier

## Classes

| Emoji | Class |
|-------|-------|
| 🏙️ | Buildings |
| 🌲 | Forest |
| 🧊 | Glacier |
| ⛰️ | Mountain |
| 🌊 | Sea |
| 🛣️ | Street |

## Project Structure

```
intel-image-classifier/
├── app.py                  # Flask web server
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   ├── __init__.py
│   └── cnn1.py             # PyTorch CNN architecture
├── ml/
│   ├── __init__.py
│   ├── dataset.py          # Dataset & DataLoader helpers
│   ├── train.py            # PyTorch training loop
│   ├── evaluate.py         # Evaluation with per-class accuracy
│   └── main.py             # Training entry point (PyTorch & TensorFlow)
├── static/
│   ├── css/style.css
│   └── uploads/.gitkeep
└── templates/
    └── index.html
```

> **Model weights** (`*.pth`, `*.keras`) are excluded from Git (they are large binary files).  
> Download them from the HuggingFace Space or train your own with the instructions below.

## Installation

```bash
git clone https://github.com/<your-username>/intel-image-classifier.git
cd intel-image-classifier

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Dataset

Download the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) from Kaggle and place it as:

```
intel-image-classifier/
└── archive/
    ├── seg_train/seg_train/
    │   ├── buildings/
    │   ├── forest/
    │   └── ...
    └── seg_test/seg_test/
        ├── buildings/
        └── ...
```

## Training

```bash
# PyTorch (saves models/model.pth)
python ml/main.py --model pytorch --firstname YourName --epochs 20

# TensorFlow (saves models/YourName_model.keras)
python ml/main.py --model tensorflow --firstname YourName --epochs 20

# Custom data directory
python ml/main.py --model pytorch --data_dir /path/to/data/root
```

## Running the Web App

```bash
python app.py
```

Open http://localhost:5000 in your browser.

The app will look for:
- `models/Julianna_model.pth` for PyTorch
- `models/Julianna_model.keras` for TensorFlow

## Deployment

### Render / Railway / Fly.io

The app reads a `PORT` environment variable automatically. Use gunicorn for production:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

### HuggingFace Spaces

Push to a Space with `SDK: gradio` or `SDK: docker`. See the live demo above.

## Dependencies

| Package | Version |
|---------|---------|
| Flask | 3.0.3 |
| PyTorch | 2.2.2 |
| TorchVision | 0.17.2 |
| TensorFlow | 2.19.0 |
| Pillow | 10.4.0 |
| NumPy | 1.26.4 |
| Gunicorn | 22.0.0 |
=======
# image_intel
image classification on the world


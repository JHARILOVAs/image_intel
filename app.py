import os
import io
import sys

import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from torchvision import transforms

# ─────────────────────────────────────────────
# PATH SETUP  (ensures `models` package is importable)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

CLASS_EMOJI = {
    'buildings': '🏙️',
    'forest':    '🌲',
    'glacier':   '🧊',
    'mountain':  '⛰️',
    'sea':       '🌊',
    'street':    '🛣️',
}

PYTORCH_MODEL_PATH    = os.path.join(BASE_DIR, 'models', 'Julianna_model.pth')
TENSORFLOW_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Julianna_model.keras')

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp'}

# ─────────────────────────────────────────────
# LAZY MODEL LOADING
# ─────────────────────────────────────────────
_pytorch_model  = None
_pytorch_device = None
_tensorflow_model = None


def load_pytorch_model():
    try:
        from models.cnn1 import CNN1

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = CNN1(num_classes=6)

        if not os.path.exists(PYTORCH_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {PYTORCH_MODEL_PATH}")

        checkpoint = torch.load(PYTORCH_MODEL_PATH, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        print(f"[PyTorch] Model loaded on {device}")
        return model, device

    except Exception as exc:
        print(f"[PyTorch ERROR] {exc}")
        return None, None


def load_tensorflow_model():
    try:
        import tensorflow as tf

        if not os.path.exists(TENSORFLOW_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {TENSORFLOW_MODEL_PATH}")

        model = tf.keras.models.load_model(TENSORFLOW_MODEL_PATH)
        print("[TensorFlow] Model loaded")
        return model

    except Exception as exc:
        print(f"[TensorFlow ERROR] {exc}")
        return None


def get_pytorch_model():
    global _pytorch_model, _pytorch_device
    if _pytorch_model is None:
        _pytorch_model, _pytorch_device = load_pytorch_model()
    return _pytorch_model, _pytorch_device


def get_tensorflow_model():
    global _tensorflow_model
    if _tensorflow_model is None:
        _tensorflow_model = load_tensorflow_model()
    return _tensorflow_model


# ─────────────────────────────────────────────
# TRANSFORMS & PREDICTION HELPERS
# ─────────────────────────────────────────────
_transform_pytorch = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def _predict_pytorch(image_bytes, model, device):
    image  = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = _transform_pytorch(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        idx     = torch.argmax(probs).item()

    return CLASSES[idx], float(probs[idx]) * 100, probs.cpu().numpy().tolist()


def _predict_tensorflow(image_bytes, model):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((228, 228))
    arr   = np.array(image, dtype=np.float32) / 255.0
    arr   = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(probs))

    return CLASSES[idx], float(probs[idx]) * 100, probs.tolist()


def _allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    pytorch_ok    = os.path.exists(PYTORCH_MODEL_PATH)
    tensorflow_ok = os.path.exists(TENSORFLOW_MODEL_PATH)
    return render_template('index.html', pytorch_ok=pytorch_ok, tensorflow_ok=tensorflow_ok)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image received'}), 400

    file       = request.files['image']
    model_type = request.form.get('model', 'pytorch')

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if not _allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type. Use JPG, PNG, or WebP.'}), 400

    image_bytes = file.read()

    try:
        if model_type == 'pytorch':
            model, device = get_pytorch_model()
            if model is None:
                return jsonify({'error': 'PyTorch model unavailable. Check server logs.'}), 503
            label, confidence, all_probs = _predict_pytorch(image_bytes, model, device)

        elif model_type == 'tensorflow':
            model = get_tensorflow_model()
            if model is None:
                return jsonify({'error': 'TensorFlow model unavailable. Check server logs.'}), 503
            label, confidence, all_probs = _predict_tensorflow(image_bytes, model)

        else:
            return jsonify({'error': f'Unknown model type: {model_type}'}), 400

        return jsonify({
            'label':      label,
            'emoji':      CLASS_EMOJI.get(label, ''),
            'confidence': round(confidence, 2),
            'all_probs': {
                CLASSES[i]: round(all_probs[i] * 100, 2)
                for i in range(len(CLASSES))
            },
            'model_used': model_type,
        })

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

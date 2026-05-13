import sys
import os
import argparse
import torch

# ─────────────────────────────────────────────
# FIX PROJECT ROOT PATH
# main.py lives in <project>/ml/
# PROJECT_ROOT = <project>/
# ─────────────────────────────────────────────
ML_DIR       = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ML_DIR)

# Add project root to sys.path so `models` and `ml` packages are importable
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.dataset import get_dataloaders
from models.cnn1 import CNN1
from ml.train import train
from ml.evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Train Intel Image Classifier")
    parser.add_argument("--model",      type=str,   choices=["pytorch", "tensorflow"], required=True)
    parser.add_argument("--firstname",  type=str,   default="model")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--data_dir",   type=str,   default=None,
                        help="Root directory containing archive/ (default: project root)")
    return parser.parse_args()


def run_pytorch(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_dir or PROJECT_ROOT

    print(f"[PyTorch] Training on device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        data_root=data_root
    )

    model      = CNN1(num_classes=6)
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    save_path  = os.path.join(models_dir, f"{args.firstname}_model.pth")

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=1e-4,
        device=device,
        save_path=save_path,
    )

    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("[PyTorch] Evaluating best model...")
    evaluate(model, test_loader, device)


def run_tensorflow(args):
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks

    data_root = args.data_dir or PROJECT_ROOT
    train_dir = os.path.join(data_root, "archive", "seg_train", "seg_train")
    test_dir  = os.path.join(data_root, "archive", "seg_test",  "seg_test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(228, 228), batch_size=args.batch_size, label_mode="int"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=(228, 228), batch_size=args.batch_size, label_mode="int"
    )

    norm     = layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (norm(x), y))
    val_ds   = val_ds.map(lambda x, y: (norm(x), y))

    model = models.Sequential([
        layers.Input(shape=(228, 228, 3)),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(1024, activation="relu"), layers.Dropout(0.2),
        layers.Dense(128,  activation="relu"), layers.Dropout(0.2),
        layers.Dense(6, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, f"{args.firstname}_model.keras")

    cb_list = [
        callbacks.ModelCheckpoint(save_path, save_best_only=True, monitor="val_loss"),
        callbacks.EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True),
    ]

    print(f"[TensorFlow] Training → {save_path}")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cb_list)
    print(f"[TensorFlow] Model saved → {save_path}")


if __name__ == "__main__":
    args = parse_args()
    if args.model == "pytorch":
        run_pytorch(args)
    elif args.model == "tensorflow":
        run_tensorflow(args)

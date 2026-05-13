import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(model, train_loader, val_loader, epochs, lr, weight_decay, device, save_path="best_model.pth"):
    """
    Train a PyTorch model with Adam + ReduceLROnPlateau scheduler.

    Returns
    -------
    dict  Training history with keys: train_loss, train_acc, val_loss, val_acc
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    model.to(device)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):

        # ── TRAIN ──────────────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        # ── VALIDATION ─────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct = 0.0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        # ── METRICS ────────────────────────────────────────────────────────────
        n_train = max(len(train_loader), 1)
        n_val   = max(len(val_loader),   1)

        avg_train_loss = train_loss / n_train
        avg_val_loss   = val_loss   / n_val
        train_acc      = 100.0 * train_correct / max(len(train_loader.dataset), 1)
        val_acc        = 100.0 * val_correct   / max(len(val_loader.dataset),   1)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(
            f"Epoch [{epoch + 1:>3}/{epochs}] "
            f"| Train Loss: {avg_train_loss:.4f}  Acc: {train_acc:.2f}% "
            f"| Val Loss: {avg_val_loss:.4f}  Acc: {val_acc:.2f}%"
        )

        scheduler.step(avg_val_loss)

        # ── SAVE BEST ──────────────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch':                epoch + 1,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':             best_val_loss,
                'val_acc':              val_acc,
            }, save_path)
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")

    print(f"\nTraining complete. Best model saved at: {save_path}")
    return history

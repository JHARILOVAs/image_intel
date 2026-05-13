import torch
import torch.nn as nn

CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


def evaluate(model, test_loader, device):
    """
    Evaluate a PyTorch model on the test set.

    Returns
    -------
    float  Overall accuracy (%)
    """
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.eval()

    test_loss    = 0.0
    test_correct = 0
    class_correct = [0] * len(CLASSES)
    class_total   = [0] * len(CLASSES)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs     = model(images)
            loss        = criterion(outputs, labels)
            predictions = outputs.argmax(1)

            test_loss    += loss.item()
            test_correct += (predictions == labels).sum().item()

            for label, pred in zip(labels, predictions):
                class_correct[label] += int(pred == label)
                class_total[label]   += 1

    avg_loss = test_loss / max(len(test_loader), 1)
    accuracy = 100.0 * test_correct / max(len(test_loader.dataset), 1)

    print(f"\nTest results:")
    print(f"  Loss     : {avg_loss:.4f}")
    print(f"  Accuracy : {accuracy:.2f}%")
    print(f"\nPer-class accuracy:")
    for i, cls in enumerate(CLASSES):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            print(f"  {cls:<12}: {acc:.2f}%  ({class_correct[i]}/{class_total[i]})")

    return accuracy

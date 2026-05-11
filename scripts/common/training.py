from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_loss_curve(record_loss_train, record_loss_test, save_path=None):
    plt.plot(range(len(record_loss_train)), record_loss_train, label="Train")
    plt.plot(range(len(record_loss_test)), record_loss_test, label="Test")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    max_loss = max(max(record_loss_train, default=0), max(record_loss_test, default=0))
    upper = max_loss * 1.05 if max_loss > 0 else 1.0
    plt.ylim(0, upper)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"loss curve saved to: {save_path}")

    plt.show()


def load_training_results(model_class, checkpoint_path, device="cuda"):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"保存済みの学習結果がありません: {checkpoint_path}. 先に学習セルを1回だけ実行してください。"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    net = model_class().to(device)
    net.load_state_dict(checkpoint["model_state_dict"])

    return net, checkpoint["record_loss_train"], checkpoint["record_loss_test"]


def save_training_results(net, record_loss_train, record_loss_test, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "record_loss_train": record_loss_train,
            "record_loss_test": record_loss_test,
        },
        checkpoint_path,
    )
    print(f"checkpoint saved to: {checkpoint_path}")


def calculate_accuracy(net, data_loader, device="cuda", num_classes=10):
    tp_per_class = torch.zeros(num_classes, dtype=torch.long)
    fp_per_class = torch.zeros(num_classes, dtype=torch.long)
    fn_per_class = torch.zeros(num_classes, dtype=torch.long)

    correct = 0
    total = 0

    net.eval()
    with torch.no_grad():
        for _, (x, t) in enumerate(data_loader):
            x, t = x.to(device), t.to(device)
            y = net(x)
            pred = y.argmax(1)

            correct += (pred == t).sum().item()
            total += len(x)

            for c in range(num_classes):
                tp_per_class[c] += ((pred == c) & (t == c)).sum().cpu()
                fp_per_class[c] += ((pred == c) & (t != c)).sum().cpu()
                fn_per_class[c] += ((pred != c) & (t == c)).sum().cpu()

    accuracy = correct / total
    precision_per_class = tp_per_class.float() / (tp_per_class.float() + fp_per_class.float() + 1e-12)
    recall_per_class = tp_per_class.float() / (tp_per_class.float() + fn_per_class.float() + 1e-12)

    return {
        "accuracy": accuracy,
        "precision": precision_per_class.mean().item(),
        "recall": recall_per_class.mean().item(),
    }

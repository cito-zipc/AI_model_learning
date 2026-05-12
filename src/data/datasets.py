import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100



def build_cifar_transforms():
    affine = transforms.RandomAffine([-30, 30], scale=(0.8, 1.2))
    shift = transforms.RandomAffine((0, 0), translate=(0.5, 0.5))
    flip = transforms.RandomHorizontalFlip(p=0.5)
    # erasing = transforms.RandomErasing(p=0.5)
    erasing = transforms.RandomErasing(p=0.4)
    # normalize = transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    normalize = transforms.Normalize(
    mean=(0.5071, 0.4867, 0.4408),
    std=(0.2675, 0.2565, 0.2761),
)
    to_tensor = transforms.ToTensor()

    transform_train = transforms.Compose([to_tensor, normalize])
    transform_train_erasing = transforms.Compose([to_tensor, erasing, normalize])
    transform_train_random = transforms.Compose([
        transforms.RandomApply([affine], p=0.5),
        transforms.RandomApply([shift], p=0.5),
        transforms.RandomApply([flip], p=0.5),
        to_tensor,
        transforms.RandomApply([erasing], p=0.5),
        normalize,
    ])
    transform_test = transforms.Compose([to_tensor, normalize])

    return {
        "train": transform_train,
        "train_erasing": transform_train_erasing,
        "train_random": transform_train_random,
        "test": transform_test,
    }


def create_cifar_datasets(data_dir="data/", transforms_map=None):
    if transforms_map is None:
        transforms_map = build_cifar_transforms()

    return {
        "cifar10_train": CIFAR10(str(data_dir), train=True, download=True, transform=transforms_map["train"]),
        "cifar10_train_erasing": CIFAR10(str(data_dir), train=True, download=True, transform=transforms_map["train_erasing"]),
        "cifar10_train_random": CIFAR10(str(data_dir), train=True, download=True, transform=transforms_map["train_random"]),
        "cifar10_test": CIFAR10(str(data_dir), train=False, download=True, transform=transforms_map["test"]),
        "cifar100_train": CIFAR100(str(data_dir), train=True, download=True, transform=transforms_map["train"]),
        "cifar100_train_erasing": CIFAR100(str(data_dir), train=True, download=True, transform=transforms_map["train_erasing"]),
        "cifar100_train_random": CIFAR100(str(data_dir), train=True, download=True, transform=transforms_map["train_random"]),
        "cifar100_test": CIFAR100(str(data_dir), train=False, download=True, transform=transforms_map["test"]),
    }





def visualize_preprocessed_images(data_loader, title="Preprocessed Images (3 each)"):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    images = next(iter(data_loader))[0]
    for i in range(3):
        img = images[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        axes[i].imshow(img)
        axes[i].set_title(f"Train [{i}]")
        axes[i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

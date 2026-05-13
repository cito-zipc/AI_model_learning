from torch.utils.data import DataLoader
from lightning import LightningDataModule
from .datasets import create_cifar_datasets, build_cifar_transforms


class CIFAR100DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        use_erasing: bool = False,
        use_random: bool = False,
        use_crop: bool = False,
        num_workers: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_erasing = use_erasing
        self.use_random = use_random
        self.use_crop = use_crop
        self.num_workers = num_workers

    def setup(self, stage=None):
        tfm = build_cifar_transforms()  # transforms_mapは使わず内部で生成
        datasets = create_cifar_datasets(
            data_dir=self.data_dir,
            transforms_map=tfm  # build済みのtfmをそのまま渡す
        )
        if self.use_random:
            self.train_dataset = datasets["cifar100_train_random"]
        elif self.use_erasing:
            self.train_dataset = datasets["cifar100_train_erasing"]
        elif self.use_crop:
            self.train_dataset = datasets["cifar100_train_crop"]
        else:
            self.train_dataset = datasets["cifar100_train"]
        self.test_dataset = datasets["cifar100_test"]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
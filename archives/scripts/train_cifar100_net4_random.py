#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CIFAR100 + Net4 の学習スクリプト。"""

import torch
import torch.nn as nn
from torch import optim
from pathlib import Path

from common.datasets import create_cifar_dataloaders
from archives.scripts.common.paths import CHECKPOINTS_DIR, FIGURES_DIR, PROJECT_ROOT
from archives.scripts.common.training import (
    calculate_accuracy,
    load_training_results,
    plot_loss_curve,
    save_training_results,
)


BATCH_SIZE = 64
NUM_EPOCHS = 50

dataloaders = create_cifar_dataloaders(batch_size=BATCH_SIZE)
train_loader_random_100 = dataloaders["train_loader_random_100"]
test_loader_100 = dataloaders["test_loader_100"]


# ====================================================================
# モデル定義
# ====================================================================

class Net4(nn.Module):
    """
    CIFAR100用のニューラルネットワーク
    Conv: 3→32→64, カーネル3x3
    FC: 64x6x6→256→100, クラス数100
    """
    def __init__(self):
        super().__init__()
        # 畳み込み層（入力チャンネル数、フィルタ数、フィルタサイズ）
        self.conv1 = nn.Conv2d(3, 32, 3)     # フィルタ数6->32、フィルタサイズ5->3
        self.conv2 = nn.Conv2d(32, 64, 3)    # フィルタ数16->64、フィルタサイズを5->3

        # 活性化関数ReLU
        self.relu = nn.ReLU()

        # プーリング層（領域のサイズ、領域の感覚）
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全結合層
        self.fc1 = nn.Linear(64*6*6, 256)

        # ドロップアウト（ドロップアウト率）
        self.dropout = nn.Dropout(p=0.5)

        # 全結合層
        self.fc2 = nn.Linear(256, 100)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ====================================================================
# 学習率を小さくして学習
# ====================================================================
# - データセット：CIFAR100
# - 前処理：複数ノイズをランダム付与
# - モデル：Net4（Conv: 3→32→64, カーネル3x3, FC: 64x6x6→256→100, クラス数100）
# - エポック数：50
# - 学習率：0.001（デフォルト）

def train_model():
    """モデルを訓練"""
    net = Net4()
    net.cuda()  # GPU対応

    loss_fnc = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    record_loss_train = []
    record_loss_test = []
    
    for i in range(NUM_EPOCHS):
        net.train()  # 訓練モード
        loss_train = 0
        
        for j, (x, t) in enumerate(train_loader_random_100):
            x, t = x.cuda(), t.cuda()
            y = net(x)
            loss = loss_fnc(y, t)
            loss_train += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        loss_train /= j + 1
        record_loss_train.append(loss_train)

        net.eval()
        loss_test = 0
        for j, (x, t) in enumerate(test_loader_100):
            x, t = x.cuda(), t.cuda()
            y = net(x)
            loss = loss_fnc(y, t)
            loss_test += loss.item()
        
        loss_test /= j + 1
        record_loss_test.append(loss_test)

        if i % 1 == 0:
            print("Epoch:", i, "Loss_Train:", loss_train, "Loss_Test:", loss_test)

    return net, record_loss_train, record_loss_test


# ====================================================================
# メイン処理
# ====================================================================

if __name__ == "__main__":
    # スクリプト名から自動的に名前を生成
    script_name = Path(__file__).stem  # train_cifar100_net4_random
    name_suffix = script_name.replace('train_', '')  # cifar100_net4_random
    
    checkpoint_path = CHECKPOINTS_DIR / f"{name_suffix}_checkpoint.pt"
    legacy_checkpoint_path = PROJECT_ROOT / "results" / f"{name_suffix}_checkpoint.pt"

    if (not checkpoint_path.exists()) and legacy_checkpoint_path.exists():
        print(f"旧チェックポイントを移行します: {legacy_checkpoint_path} -> {checkpoint_path}")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_checkpoint_path.replace(checkpoint_path)

    if checkpoint_path.exists():
        print(f"チェックポイントを読み込みます: {checkpoint_path}")
        net, record_loss_train, record_loss_test = load_training_results(
            Net4, checkpoint_path, device="cuda"
        )
    else:
        # モデルの訓練
        print("モデルの訓練を開始します...")
        net, record_loss_train, record_loss_test = train_model()
        save_training_results(net, record_loss_train, record_loss_test, checkpoint_path)

    # 結果出力
    print("\n学習結果を表示します...")
    plot_loss_curve(record_loss_train, record_loss_test, save_path=FIGURES_DIR / f"{name_suffix}.png")
    
    metrics = calculate_accuracy(net, test_loader_100, num_classes=100)
    print(f"accuracy : {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall   : {metrics['recall']:.4f}")

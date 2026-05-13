import torch.nn as nn

# ====================================================================
# モデル定義
# ====================================================================
    
class Net5(nn.Module):
    """
    CIFAR100用のニューラルネットワーク
    Conv: 3→32→64→128, カーネル3x3, 3回の畳み込みとプーリング
    FC: 128x6x6→256→100, クラス数100
    バッチ正規化有効
    """
    def __init__(self):
        super().__init__()
        # 畳み込み層（入力数、フィルタ数、フィルタサイズ）
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)     
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)  
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)   

        # バッチ正規化層
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)

        # 活性化関数ReLU
        self.relu = nn.ReLU()

        # プーリング層（領域のサイズ、領域の感覚）
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全結合層
        self.fc1 = nn.Linear(256*4*4, 256)

        # ドロップアウト（ドロップアウト率）
        self.dropout = nn.Dropout(p=0.5)

        # 全結合層
        self.fc2 = nn.Linear(256, 100)

    def forward(self, x):
        # 1回目の畳み込み、活性化、バッチ正規化、プーリング
        x = self.conv1(x)   # 3x32x32 → 64x32x32
        x = self.relu(x)    # 64x32x32 → 64x32x32
        x = self.batch_norm1(x) # 64x32x32 → 64x32x32
        x = self.pool(x)    # 64x32x32 → 64x16x16

        # 2回目の畳み込み、活性化、バッチ正規化、プーリング
        x = self.conv2(x)   # 64x16x16 → 128x16x16
        x = self.relu(x)    # 128x16x16 → 128x16x16
        x = self.batch_norm2(x) # 128x16x16 → 128x16x16
        x = self.pool(x)    # 128x16x16 → 128x8x8

        # 3回目の畳み込み、活性化、バッチ正規化、プーリング
        x = self.conv3(x)   # 128x8x8 → 256x8x8
        x = self.relu(x)    # 256x8x8 → 256x8x8
        x = self.batch_norm3(x) # 256x8x8 → 256x8x8
        x = self.pool(x)    # 256x8x8 → 256x4x4

        # 全結合層に入力するためにフラット化
        x = x.view(x.size(0), -1)
        x = self.fc1(x)     # 256*4*4 → 256
        x = self.relu(x)    # 256 → 256
        x = self.dropout(x)
        x = self.fc2(x)
        return x


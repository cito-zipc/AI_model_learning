import torch.nn as nn

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


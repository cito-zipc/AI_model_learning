# AI Model Learning — CIFAR-100 実験プロジェクト

PyTorch Lightning + Hydra を用いた CIFAR-100 の画像分類実験プロジェクトです。

## プロジェクト構成

```
src/train.py          # メイン学習スクリプト
configs/              # Hydra 設定ファイル
  experiment/         # 実験ごとの設定
  model/              # モデル設定
  data/               # データセット設定
  logger/             # ロガー設定 (MLflow / TensorBoard)
data/cifar/           # CIFAR データセット
logs/mlflow/          # MLflow ログ出力先
```

## 環境セットアップ

```bash
conda activate ml_gpu311
pip install -r requirements.txt
```

## 学習方法

学習は `src/train.py` を Hydra 経由で実行します。  
`experiment` オプションで実験設定ファイルを切り替えられます。

### 基本実行（デフォルト設定）

```bash
python src/train.py experiment=cifar100_net4
```

### オーバーライド例

```bash
# エポック数を変更
python src/train.py experiment=cifar100_net4 trainer.max_epochs=50

# GPU を使用
python src/train.py experiment=cifar100_net4 trainer=gpu

# テストのみ実行
python src/train.py experiment=cifar100_net4 train=False test=True
```

## MLflow の起動方法

学習ログは SQLite (`mlruns.db`) に保存されます。  
以下のコマンドで MLflow UI を起動して結果を確認できます。

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

起動後、ブラウザで http://localhost:5000 にアクセスしてください。

> ポートを変更したい場合:
> ```bash
> mlflow ui --backend-store-uri sqlite:///mlruns.db --port 8080
> ```

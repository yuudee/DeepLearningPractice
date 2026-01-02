# -*- coding: utf-8 -*-
"""
MNIST分類のためのCNN（畳み込みニューラルネットワーク）モデル
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    MNIST分類のための浅いCNNモデル

    構造:
        - Conv1: 1チャンネル -> 32チャンネル, 3x3カーネル
        - MaxPool: 2x2
        - Conv2: 32チャンネル -> 64チャンネル, 3x3カーネル
        - MaxPool: 2x2
        - FC1: 1600 -> 128
        - FC2: 128 -> 10 (クラス数)
    """

    def __init__(self, num_classes=10, dropout_rate=0.25):
        """
        Args:
            num_classes (int): 出力クラス数（デフォルト: 10）
            dropout_rate (float): ドロップアウト率（デフォルト: 0.25）
        """
        super(SimpleCNN, self).__init__()

        # 畳み込み層
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # プーリング層
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ドロップアウト
        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate * 2)

        # 全結合層
        # 入力サイズ: 28x28 -> conv1+pool -> 14x14 -> conv2+pool -> 7x7
        # 64チャンネル x 7 x 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        順伝播

        Args:
            x: 入力テンソル (batch_size, 1, 28, 28)

        Returns:
            出力テンソル (batch_size, num_classes)
        """
        # 第1畳み込みブロック
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        # 第2畳み込みブロック
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # 全結合層
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


class DeepCNN(nn.Module):
    """
    MNIST分類のためのやや深いCNNモデル（比較用）

    構造:
        - Conv1: 1 -> 32, 3x3
        - Conv2: 32 -> 32, 3x3
        - MaxPool: 2x2
        - Conv3: 32 -> 64, 3x3
        - Conv4: 64 -> 64, 3x3
        - MaxPool: 2x2
        - FC1: 1024 -> 256
        - FC2: 256 -> 10
    """

    def __init__(self, num_classes=10, dropout_rate=0.25):
        super(DeepCNN, self).__init__()

        # 第1畳み込みブロック
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        # 第2畳み込みブロック
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate * 2)

        # 28x28 -> conv1,conv2+pool -> 14x14 -> conv3,conv4+pool -> 7x7
        # valid padding使用時: 28->26->24->12->10->8->4 = 4x4x64 = 1024
        # same padding使用時: 28->28->28->14->14->14->7 = 7x7x64 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 第1ブロック
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)

        # 第2ブロック
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout1(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # 全結合層
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


def get_cnn_model(model_type='simple', num_classes=10, dropout_rate=0.25):
    """
    CNNモデルを取得するファクトリ関数

    Args:
        model_type (str): モデルタイプ ('simple' または 'deep')
        num_classes (int): 出力クラス数
        dropout_rate (float): ドロップアウト率

    Returns:
        nn.Module: CNNモデル
    """
    if model_type == 'simple':
        return SimpleCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_type == 'deep':
        return DeepCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'simple' or 'deep'.")


def count_parameters(model):
    """
    モデルの学習可能パラメータ数をカウント

    Args:
        model (nn.Module): PyTorchモデル

    Returns:
        int: パラメータ数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -*- coding: utf-8 -*-
"""
MNISTデータを読み込むためのユーティリティ関数
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_mnist(batch_size=64, root='./data', train=True):
    """
    MNISTデータセットを読み込む関数

    Args:
        batch_size (int): バッチサイズ（デフォルト: 64）
        root (str): データを保存するディレクトリ（デフォルト: './data'）
        train (bool): Trueの場合は訓練データ、Falseの場合はテストデータを読み込む

    Returns:
        DataLoader: MNISTデータセットのDataLoader
    """
    # データの前処理（正規化）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNISTの平均と標準偏差
    ])

    # MNISTデータセットをダウンロードして読み込む
    dataset = datasets.MNIST(
        root=root,
        train=train,
        download=True,
        transform=transform
    )

    # DataLoaderを作成
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train  # 訓練データの場合はシャッフル
    )

    return dataloader


def get_mnist_data_loaders(batch_size=64, root='./data'):
    """
    訓練データとテストデータの両方のDataLoaderを取得する関数

    Args:
        batch_size (int): バッチサイズ（デフォルト: 64）
        root (str): データを保存するディレクトリ（デフォルト: './data'）

    Returns:
        tuple: (train_loader, test_loader) のタプル
    """
    train_loader = load_mnist(batch_size=batch_size, root=root, train=True)
    test_loader = load_mnist(batch_size=batch_size, root=root, train=False)

    return train_loader, test_loader

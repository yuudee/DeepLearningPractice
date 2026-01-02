# -*- coding: utf-8 -*-
"""
Adadelta（Adaptive Learning Rate Method）用のオプティマイザ作成関数
"""
import torch.optim as optim
from trainer import train_with_optimizer


def create_adadelta_optimizer(learning_rate=1.0, rho=0.9, eps=1e-6, weight_decay=0.0):
    """
    Adadeltaオプティマイザを作成する関数

    Args:
        learning_rate (float): 学習率（デフォルト: 1.0、Adadeltaでは通常1.0）
        rho (float): 減衰率（デフォルト: 0.9）
        eps (float): 数値安定性のための小さな値（デフォルト: 1e-6）
        weight_decay (float): 重み減衰（L2正則化）係数（デフォルト: 0.0）

    Returns:
        callable: オプティマイザを返す関数（model.parameters()を引数に取る）
    """
    def optimizer_fn(params):
        return optim.Adadelta(
            params,
            lr=learning_rate,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay
        )
    return optimizer_fn


def train_with_adadelta(model, train_loader, num_epochs=10, learning_rate=1.0,
                        rho=0.9, eps=1e-6, weight_decay=0.0, device=None,
                        verbose=True, epoch_callback=None):
    """
    Adadeltaを用いてモデルを学習する関数（後方互換性のため）

    Args:
        model (nn.Module): 学習するモデル
        train_loader (DataLoader): 訓練データのDataLoader
        num_epochs (int): エポック数（デフォルト: 10）
        learning_rate (float): 学習率（デフォルト: 1.0、Adadeltaでは通常1.0）
        rho (float): 減衰率（デフォルト: 0.9）
        eps (float): 数値安定性のための小さな値（デフォルト: 1e-6）
        weight_decay (float): 重み減衰（L2正則化）係数（デフォルト: 0.0）
        device (torch.device): 使用するデバイス（Noneの場合は自動選択）
        verbose (bool): 進捗を表示するかどうか（デフォルト: True）
        epoch_callback (callable): 各エポック後に呼び出されるコールバック関数

    Returns:
        list: 各エポックの損失のリスト
    """
    optimizer_fn = create_adadelta_optimizer(
        learning_rate=learning_rate,
        rho=rho,
        eps=eps,
        weight_decay=weight_decay
    )
    return train_with_optimizer(
        model=model,
        train_loader=train_loader,
        optimizer_fn=optimizer_fn,
        num_epochs=num_epochs,
        device=device,
        verbose=verbose,
        epoch_callback=epoch_callback
    )

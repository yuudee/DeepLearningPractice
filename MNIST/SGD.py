# -*- coding: utf-8 -*-
"""
SGD（Stochastic Gradient Descent）用のオプティマイザ作成関数
"""
import torch.optim as optim
from trainer import train_with_optimizer


def create_sgd_optimizer(learning_rate=0.01, momentum=0.0, nesterov=False, weight_decay=0.0):
    """
    SGDオプティマイザを作成する関数

    Args:
        learning_rate (float): 学習率（デフォルト: 0.01）
        momentum (float): モーメンタム係数（デフォルト: 0.0、0.9が一般的）
        nesterov (bool): ネステロフの加速勾配を使用するかどうか（デフォルト: False）
        weight_decay (float): L2正則化（重み減衰）係数（デフォルト: 0.0）

    Returns:
        callable: オプティマイザを返す関数（model.parameters()を引数に取る）
    """
    def optimizer_fn(params):
        return optim.SGD(
            params,
            lr=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    return optimizer_fn


def train_with_sgd(model, train_loader, num_epochs=10, learning_rate=0.01,
                   momentum=0.0, nesterov=False, weight_decay=0.0, device=None,
                   verbose=True, epoch_callback=None):
    """
    SGDを用いてモデルを学習する関数（後方互換性のため）

    Args:
        model (nn.Module): 学習するモデル
        train_loader (DataLoader): 訓練データのDataLoader
        num_epochs (int): エポック数（デフォルト: 10）
        learning_rate (float): 学習率（デフォルト: 0.01）
        momentum (float): モーメンタム係数（デフォルト: 0.0、0.9が一般的）
        nesterov (bool): ネステロフの加速勾配を使用するかどうか（デフォルト: False）
        weight_decay (float): L2正則化（重み減衰）係数（デフォルト: 0.0）
        device (torch.device): 使用するデバイス（Noneの場合は自動選択）
        verbose (bool): 進捗を表示するかどうか（デフォルト: True）
        epoch_callback (callable): 各エポック後に呼び出されるコールバック関数

    Returns:
        list: 各エポックの損失のリスト
    """
    optimizer_fn = create_sgd_optimizer(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov,
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

"""
Adagrad（Adaptive Gradient Algorithm）用のオプティマイザー作成関数
"""
import torch.optim as optim
from trainer import train_with_optimizer


def create_adagrad_optimizer(learning_rate=0.01, lr_decay=0.0, weight_decay=0.0, eps=1e-10):
    """
    Adagradオプティマイザーを作成する関数
    
    Args:
        learning_rate (float): 学習率（デフォルト: 0.01）
        lr_decay (float): 学習率の減衰率（デフォルト: 0.0）
        weight_decay (float): 重み減衰（L2正則化）係数（デフォルト: 0.0）
        eps (float): 数値安定性のための小さな値（デフォルト: 1e-10）
    
    Returns:
        callable: オプティマイザーを返す関数（model.parameters()を引数に取る）
    """
    def optimizer_fn(params):
        return optim.Adagrad(
            params,
            lr=learning_rate,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            eps=eps
        )
    return optimizer_fn


def train_with_adagrad(model, train_loader, num_epochs=10, learning_rate=0.01,
                       lr_decay=0.0, weight_decay=0.0, eps=1e-10, device=None,
                       verbose=True, epoch_callback=None):
    """
    Adagradを用いてモデルを学習する関数（後方互換性のため）
    
    Args:
        model (nn.Module): 学習するモデル
        train_loader (DataLoader): 訓練データのDataLoader
        num_epochs (int): エポック数（デフォルト: 10）
        learning_rate (float): 学習率（デフォルト: 0.01）
        lr_decay (float): 学習率の減衰率（デフォルト: 0.0）
        weight_decay (float): 重み減衰（L2正則化）係数（デフォルト: 0.0）
        eps (float): 数値安定性のための小さな値（デフォルト: 1e-10）
        device (torch.device): 使用するデバイス（Noneの場合は自動選択）
        verbose (bool): 進捗を表示するかどうか（デフォルト: True）
        epoch_callback (callable): 各エポック後に呼び出されるコールバック関数
    
    Returns:
        list: 各エポックの損失のリスト
    """
    optimizer_fn = create_adagrad_optimizer(
        learning_rate=learning_rate,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        eps=eps
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


# -*- coding: utf-8 -*-
"""
汎用的なモデル訓練関数
"""
import torch
import torch.nn as nn
from tqdm import tqdm


def train_with_optimizer(model, train_loader, optimizer_fn, num_epochs=10,
                         device=None, verbose=True, epoch_callback=None):
    """
    汎用的なモデル訓練関数（任意のオプティマイザに対応）

    Args:
        model (nn.Module): 学習するモデル
        train_loader (DataLoader): 訓練データのDataLoader
        optimizer_fn (callable): オプティマイザを返す関数（model.parameters()を引数に取る）
        num_epochs (int): エポック数（デフォルト: 10）
        device (torch.device): 使用するデバイス（Noneの場合は自動選択）
        verbose (bool): 進捗を表示するかどうか（デフォルト: True）
        epoch_callback (callable): 各エポック後に呼び出されるコールバック関数
                                   (epoch, avg_loss, model) を引数に取る

    Returns:
        list: 各エポックの損失のリスト
    """
    # デバイスの設定
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # 損失関数とオプティマイザの設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_fn(model.parameters())

    # 損失の記録用リスト
    losses = []

    # モデルを訓練モードに設定
    model.train()

    # エポックごとの学習
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # 進捗バーの設定
        if verbose:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        else:
            pbar = train_loader

        # バッチごとの学習
        for batch_idx, (data, target) in enumerate(pbar):
            # データをデバイスに移動
            data, target = data.to(device), target.to(device)

            # 勾配をゼロにリセット
            optimizer.zero_grad()

            # 順伝播
            output = model(data)
            loss = criterion(output, target)

            # 逆伝播
            loss.backward()

            # パラメータの更新
            optimizer.step()

            # 損失の記録
            epoch_loss += loss.item()
            num_batches += 1

            # 進捗バーの更新
            if verbose:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # エポック平均損失を記録
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        # コールバック関数の呼び出し（各エポック後にテストlossを計算するなど）
        if epoch_callback is not None:
            epoch_callback(epoch, avg_loss, model)

        # コールバックがない場合のみverbose出力（コールバック内で表示するため）
        if verbose and epoch_callback is None:
            print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

    return losses

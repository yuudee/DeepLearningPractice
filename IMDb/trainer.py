# -*- coding: utf-8 -*-
"""
学習・評価のためのトレーナークラス
"""
import torch
import torch.nn as nn
import time


def binary_accuracy(predictions, labels):
    """
    バイナリ分類の正解率を計算

    Args:
        predictions: モデルの予測値
        labels: 正解ラベル

    Returns:
        正解率
    """
    # シグモイドを適用して0-1の確率に変換
    preds = torch.sigmoid(predictions)
    # 0.5を閾値として二値化
    rounded_preds = torch.round(preds)
    # 正解率を計算
    correct = (rounded_preds == labels).float()
    acc = correct.sum() / len(correct)
    return acc


def train_epoch(model, data_loader, optimizer, criterion, device, use_lengths=True):
    """
    1エポックの学習

    Args:
        model: モデル
        data_loader: データローダー
        optimizer: オプティマイザ
        criterion: 損失関数
        device: デバイス
        use_lengths: 系列長を使用するか

    Returns:
        epoch_loss, epoch_acc
    """
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch in data_loader:
        text = batch['text'].to(device)
        labels = batch['label'].to(device)
        lengths = batch['length'].to(device) if use_lengths else None

        optimizer.zero_grad()

        # 順伝播
        if use_lengths:
            predictions = model(text, lengths).squeeze(1)
        else:
            predictions = model(text).squeeze(1)

        # 損失計算
        loss = criterion(predictions, labels)

        # 正解率計算
        acc = binary_accuracy(predictions, labels)

        # 逆伝播
        loss.backward()

        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # パラメータ更新
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(data_loader), epoch_acc / len(data_loader)


def evaluate(model, data_loader, criterion, device, use_lengths=True):
    """
    評価

    Args:
        model: モデル
        data_loader: データローダー
        criterion: 損失関数
        device: デバイス
        use_lengths: 系列長を使用するか

    Returns:
        epoch_loss, epoch_acc
    """
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for batch in data_loader:
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            lengths = batch['length'].to(device) if use_lengths else None

            if use_lengths:
                predictions = model(text, lengths).squeeze(1)
            else:
                predictions = model(text).squeeze(1)

            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(data_loader), epoch_acc / len(data_loader)


def train_model(model, train_loader, val_loader, optimizer, criterion, device,
                num_epochs=10, save_callback=None, verbose=True, use_lengths=True):
    """
    モデルの学習

    Args:
        model: モデル
        train_loader: 学習用データローダー
        val_loader: 検証用データローダー
        optimizer: オプティマイザ
        criterion: 損失関数
        device: デバイス
        num_epochs: エポック数
        save_callback: エポックごとに呼ばれるコールバック関数
                       save_callback(model, epoch, train_loss, train_acc, val_loss, val_acc)
        verbose: 詳細出力
        use_lengths: 系列長を使用するか

    Returns:
        history: 学習履歴
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_time': []
    }

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        # 学習
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, use_lengths
        )

        # 検証
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, use_lengths
        )

        end_time = time.time()
        epoch_time = end_time - start_time

        # 履歴の保存
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)

        # ベストモデルの更新
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch + 1

        if verbose:
            print(f'Epoch {epoch+1:3d}/{num_epochs} | '
                  f'Time: {epoch_time:.2f}s | '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%'
                  f'{" *" if is_best else ""}')

        # コールバック呼び出し
        if save_callback is not None:
            save_callback(
                model=model,
                epoch=epoch + 1,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                is_best=is_best
            )

    if verbose:
        print(f'\nBest validation loss: {best_val_loss:.4f} at epoch {best_epoch}')

    return history


def predict(model, text_tensor, length_tensor, device, use_lengths=True):
    """
    単一のテキストに対する予測

    Args:
        model: モデル
        text_tensor: テキストのテンソル
        length_tensor: 長さのテンソル
        device: デバイス
        use_lengths: 系列長を使用するか

    Returns:
        確率値 (0-1)
    """
    model.eval()
    with torch.no_grad():
        text_tensor = text_tensor.unsqueeze(0).to(device)
        if use_lengths:
            length_tensor = length_tensor.unsqueeze(0).to(device)
            prediction = model(text_tensor, length_tensor)
        else:
            prediction = model(text_tensor)
        probability = torch.sigmoid(prediction).item()
    return probability


def predict_sentiment(model, text, vocab, max_length, device, use_lengths=True):
    """
    テキストの感情予測

    Args:
        model: モデル
        text: 入力テキスト
        vocab: 語彙
        max_length: 最大長
        device: デバイス
        use_lengths: 系列長を使用するか

    Returns:
        sentiment (str), probability (float)
    """
    from data_utils import preprocess_text

    # 前処理
    text = preprocess_text(text)

    # エンコード
    encoded = vocab.encode(text)

    # 長さの調整
    if len(encoded) > max_length:
        encoded = encoded[:max_length]
    length = len(encoded)

    # パディング
    if len(encoded) < max_length:
        encoded = encoded + [vocab.pad_idx] * (max_length - len(encoded))

    # テンソル化
    text_tensor = torch.tensor(encoded, dtype=torch.long)
    length_tensor = torch.tensor(length, dtype=torch.long)

    # 予測
    probability = predict(model, text_tensor, length_tensor, device, use_lengths)

    # 感情判定
    sentiment = 'positive' if probability >= 0.5 else 'negative'

    return sentiment, probability


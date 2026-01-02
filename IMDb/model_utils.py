# -*- coding: utf-8 -*-
"""
モデルの保存・読み込みユーティリティ
"""
import os
import torch
import pickle
from datetime import datetime


def get_save_dir(base_dir='./models'):
    """
    タイムスタンプ付きの保存ディレクトリを作成

    Args:
        base_dir: ベースディレクトリ

    Returns:
        save_dir: 保存先ディレクトリパス
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def save_checkpoint(model, optimizer, epoch, train_loss, train_acc,
                   val_loss, val_acc, vocab, model_config, save_path):
    """
    チェックポイントを保存

    Args:
        model: モデル
        optimizer: オプティマイザ
        epoch: エポック番号
        train_loss: 学習損失
        train_acc: 学習精度
        val_loss: 検証損失
        val_acc: 検証精度
        vocab: 語彙オブジェクト
        model_config: モデル設定
        save_path: 保存パス
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'model_config': model_config
    }
    torch.save(checkpoint, save_path)

    # 語彙を別途保存（チェックポイントごとに保存すると冗長なので、初回のみ）
    vocab_path = os.path.join(os.path.dirname(save_path), 'vocab.pkl')
    if not os.path.exists(vocab_path):
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)


def load_checkpoint(checkpoint_path, model, optimizer=None, device=None):
    """
    チェックポイントを読み込み

    Args:
        checkpoint_path: チェックポイントのパス
        model: モデル（state_dictをロードする先）
        optimizer: オプティマイザ（オプション）
        device: デバイス

    Returns:
        checkpoint: チェックポイント情報
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def load_vocab(vocab_path):
    """
    語彙を読み込み

    Args:
        vocab_path: 語彙ファイルのパス

    Returns:
        vocab: 語彙オブジェクト
    """
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def create_save_callback(save_dir, vocab, model_config, optimizer, verbose=True):
    """
    エポックごとの保存コールバックを作成

    Args:
        save_dir: 保存先ディレクトリ
        vocab: 語彙オブジェクト
        model_config: モデル設定
        optimizer: オプティマイザ
        verbose: 詳細出力

    Returns:
        callback: コールバック関数
    """
    best_val_loss = float('inf')

    def callback(model, epoch, train_loss, train_acc, val_loss, val_acc, is_best):
        nonlocal best_val_loss

        # エポックごとのモデルを保存
        epoch_path = os.path.join(save_dir, f'epoch_{epoch:03d}.pth')
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            vocab=vocab,
            model_config=model_config,
            save_path=epoch_path
        )

        if verbose:
            print(f'  -> Saved: {epoch_path}')

        # ベストモデルを保存
        if is_best:
            best_val_loss = val_loss
            best_path = os.path.join(save_dir, 'best_model.pth')
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                vocab=vocab,
                model_config=model_config,
                save_path=best_path
            )
            if verbose:
                print(f'  -> Saved best model: {best_path}')

    return callback


def list_saved_models(save_dir):
    """
    保存されたモデルの一覧を取得

    Args:
        save_dir: 保存ディレクトリ

    Returns:
        models: モデルファイルのリスト
    """
    if not os.path.exists(save_dir):
        return []

    models = []
    for f in sorted(os.listdir(save_dir)):
        if f.endswith('.pth'):
            models.append(os.path.join(save_dir, f))
    return models


def get_latest_model_dir(base_dir='./models'):
    """
    最新のモデルディレクトリを取得

    Args:
        base_dir: ベースディレクトリ

    Returns:
        latest_dir: 最新のディレクトリパス（なければNone）
    """
    if not os.path.exists(base_dir):
        return None

    dirs = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            dirs.append(path)

    if not dirs:
        return None

    # 名前（タイムスタンプ）でソートして最新を取得
    dirs.sort(reverse=True)
    return dirs[0]


def load_model_for_inference(model_path, model_class, device=None):
    """
    推論用にモデルを読み込み

    Args:
        model_path: モデルファイルのパス
        model_class: モデルクラス（get_model関数など）
        device: デバイス

    Returns:
        model: ロード済みモデル
        vocab: 語彙
        config: モデル設定
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # チェックポイントの読み込み
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']

    # モデルの作成
    model = model_class(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 語彙の読み込み
    vocab_path = os.path.join(os.path.dirname(model_path), 'vocab.pkl')
    vocab = load_vocab(vocab_path)

    return model, vocab, config


def save_training_config(save_dir, config):
    """
    学習設定を保存

    Args:
        save_dir: 保存ディレクトリ
        config: 設定辞書
    """
    config_path = os.path.join(save_dir, 'training_config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)

    # 人間が読める形式でも保存
    txt_path = os.path.join(save_dir, 'training_config.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Training Configuration\n")
        f.write("=" * 50 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")


def load_training_config(save_dir):
    """
    学習設定を読み込み

    Args:
        save_dir: 保存ディレクトリ

    Returns:
        config: 設定辞書
    """
    config_path = os.path.join(save_dir, 'training_config.pkl')
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    return config


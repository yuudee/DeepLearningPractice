# -*- coding: utf-8 -*-
"""
MNISTデータの学習と評価を実行するメインファイル
"""
import torch
import torch.nn as nn
import numpy as np
from utils import get_mnist_data_loaders
from SGD import train_with_sgd
from cnn_model import SimpleCNN, DeepCNN, get_cnn_model, count_parameters
from model_utils import (
    create_save_directory, save_epoch_model, save_best_model,
    load_model, load_model_for_inference, list_saved_models, get_latest_model_dir
)
try:
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("警告: sklearn, matplotlib, seabornがインストールされていません。Confusion matrixの可視化はスキップされます。")


class SimpleMNISTModel(nn.Module):
    """
    MNIST分類のためのシンプルなニューラルネットワークモデル
    """
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleMNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def calculate_test_loss(model, test_loader, criterion, device=None):
    """
    テストデータでの損失を計算する関数

    Args:
        model (nn.Module): 評価するモデル
        test_loader (DataLoader): テストデータのDataLoader
        criterion: 損失関数
        device (torch.device): 使用するデバイス（Noneの場合は自動選択）

    Returns:
        float: テストデータでの平均損失
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate_model(model, test_loader, device=None):
    """
    モデルの評価を行う関数（正解率を計算）

    Args:
        model (nn.Module): 評価するモデル
        test_loader (DataLoader): テストデータのDataLoader
        device (torch.device): 使用するデバイス（Noneの場合は自動選択）

    Returns:
        float: テストデータでの正解率
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def calculate_confusion_matrix(model, test_loader, device=None):
    """
    モデルのconfusion matrixを計算する関数

    Args:
        model (nn.Module): 評価するモデル
        test_loader (DataLoader): テストデータのDataLoader
        device (torch.device): 使用するデバイス（Noneの場合は自動選択）

    Returns:
        tuple: (confusion_matrix, all_predicted, all_targets) のタプル
            - confusion_matrix: confusion matrix（numpy配列）
            - all_predicted: すべての予測ラベルのリスト
            - all_targets: すべての正解ラベルのリスト
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    all_predicted = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            all_predicted.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    cm = confusion_matrix(all_targets, all_predicted)
    return cm, all_predicted, all_targets


def plot_confusion_matrix(model, test_loader, device=None, save_path=None,
                          class_names=None, figsize=(10, 8)):
    """
    confusion matrixを可視化する関数

    Args:
        model (nn.Module): 評価するモデル
        test_loader (DataLoader): テストデータのDataLoader
        device (torch.device): 使用するデバイス（Noneの場合は自動選択）
        save_path (str): 画像を保存するパス（Noneの場合は表示のみ）
        class_names (list): クラス名のリスト（Noneの場合は0-9）
        figsize (tuple): 図のサイズ（デフォルト: (10, 8)）
    """
    if not HAS_VISUALIZATION:
        print("可視化ライブラリがインストールされていないため、Confusion matrixを表示できません。")
        return

    cm, _, _ = calculate_confusion_matrix(model, test_loader, device)

    if class_names is None:
        class_names = [str(i) for i in range(10)]

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Sample Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

    plt.close()


def print_confusion_matrix(model, test_loader, device=None):
    """
    confusion matrixをテキスト形式で表示する関数

    Args:
        model (nn.Module): 評価するモデル
        test_loader (DataLoader): テストデータのDataLoader
        device (torch.device): 使用するデバイス（Noneの場合は自動選択）
    """
    cm, _, _ = calculate_confusion_matrix(model, test_loader, device)

    print("\n" + "=" * 50)
    print("Confusion Matrix")
    print("=" * 50)
    print("\nTrue Label (row) vs Predicted Label (column):")
    print("\n     ", end="")
    for i in range(10):
        print(f"{i:5d}", end="")
    print()

    for i in range(10):
        print(f"{i:3d}  ", end="")
        for j in range(10):
            print(f"{cm[i, j]:5d}", end="")
        print()

    # 各クラスの精度を計算
    print("\nAccuracy per class:")
    for i in range(10):
        correct = cm[i, i]
        total = cm[i, :].sum()
        accuracy = 100 * correct / total if total > 0 else 0
        print(f"  Class {i}: {correct}/{total} = {accuracy:.2f}%")

    print("=" * 50)


def train_model(model, train_loader, test_loader, optimizer_fn, num_epochs=10,
                device=None, verbose=True):
    """
    モデルを学習する汎用的な関数

    Args:
        model (nn.Module): 学習するモデル
        train_loader (DataLoader): 訓練データのDataLoader
        test_loader (DataLoader): テストデータのDataLoader
        optimizer_fn (callable): オプティマイザを返す関数（model.parameters()を引数に取る）
        num_epochs (int): エポック数（デフォルト: 10）
        device (torch.device): 使用するデバイス（Noneの場合は自動選択）
        verbose (bool): 進捗を表示するかどうか（デフォルト: True）

    Returns:
        tuple: (train_losses, test_losses) のタプル
            - train_losses: 各エポックの訓練損失のリスト
            - test_losses: 各エポックのテスト損失のリスト
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_fn(model.parameters())

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        # 訓練
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)

        # テストlossの計算
        avg_test_loss = calculate_test_loss(model, test_loader, criterion, device)
        test_losses.append(avg_test_loss)

        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

    return train_losses, test_losses


def run_training(optimizer_name, optimizer_fn, batch_size=64, num_epochs=10,
                 hidden_size=128, data_root='./data', device=None,
                 use_module=False, verbose=True, opt_info_str=""):
    """
    汎用的な学習実行関数

    Args:
        optimizer_name (str): 最適化手法の名前（表示用）
        optimizer_fn (callable): オプティマイザを作成する関数
        batch_size (int): バッチサイズ（デフォルト: 64）
        num_epochs (int): エポック数（デフォルト: 10）
        hidden_size (int): 隠れ層のサイズ（デフォルト: 128）
        data_root (str): データを保存するディレクトリ（デフォルト: './data'）
        device (torch.device): 使用するデバイス（Noneの場合は自動選択）
        use_module (bool): 専用モジュールのtrain関数を使用するかどうか（デフォルト: False）
        verbose (bool): 進捗を表示するかどうか（デフォルト: True）
        opt_info_str (str): 最適化手法の詳細情報（表示用）

    Returns:
        tuple: (model, train_losses, test_losses, test_accuracy) のタプル
    """
    print("=" * 50)
    print(f"MNIST Training Start ({optimizer_name})")
    print("=" * 50)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nLoading data...")
    train_loader, test_loader = get_mnist_data_loaders(batch_size=batch_size, root=data_root)
    print(f"Training data: {len(train_loader.dataset)} samples")
    print(f"Test data: {len(test_loader.dataset)} samples")

    print(f"\nCreating model...")
    model = SimpleMNISTModel(input_size=28*28, hidden_size=hidden_size, num_classes=10)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt_info = f"Epochs: {num_epochs}" + opt_info_str
    print(f"\nStarting training ({opt_info})...")

    if use_module:
        # 専用モジュールのtrain関数を使用する場合
        # この場合は各最適化手法のモジュールからtrain関数をインポートして使用
        # ただし、現在の実装ではtrain_modelを使用する方が簡単
        pass

    # 汎用的なtrain_model関数を使用
    train_losses, test_losses = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer_fn=optimizer_fn,
        num_epochs=num_epochs,
        device=device,
        verbose=verbose
    )

    print("\nEvaluating model...")
    test_accuracy = evaluate_model(model, test_loader, device=device)
    print(f"Test accuracy: {test_accuracy:.2f}%")

    # Confusion Matrixの表示
    print_confusion_matrix(model, test_loader, device=device)

    print("\n" + "=" * 50)
    print("Training Complete")
    print("=" * 50)

    return model, train_losses, test_losses, test_accuracy


def run_sgd_training(batch_size=64, num_epochs=10, learning_rate=0.01,
                     momentum=0.0, nesterov=False, weight_decay=0.0,
                     hidden_size=128, data_root='./data', device=None,
                     use_sgd_module=False, verbose=True):
    """
    SGDを用いてMNISTデータを学習する関数

    Args:
        batch_size (int): バッチサイズ（デフォルト: 64）
        num_epochs (int): エポック数（デフォルト: 10）
        learning_rate (float): 学習率（デフォルト: 0.01）
        momentum (float): モーメンタム係数（デフォルト: 0.0、0.9が一般的）
        nesterov (bool): ネステロフの加速勾配を使用するかどうか（デフォルト: False）
        weight_decay (float): L2正則化（重み減衰）係数（デフォルト: 0.0）
        hidden_size (int): 隠れ層のサイズ（デフォルト: 128）
        data_root (str): データを保存するディレクトリ（デフォルト: './data'）
        device (torch.device): 使用するデバイス（Noneの場合は自動選択）
        use_sgd_module (bool): SGD.pyのtrain_with_sgd関数を使用するかどうか（デフォルト: False）
        verbose (bool): 進捗を表示するかどうか（デフォルト: True）

    Returns:
        tuple: (model, train_losses, test_losses, test_accuracy) のタプル
            - model: 学習済みモデル
            - train_losses: 各エポックの訓練損失のリスト
            - test_losses: 各エポックのテスト損失のリスト
            - test_accuracy: テストデータでの正解率
    """
    from SGD import create_sgd_optimizer

    opt_info_str = f", lr: {learning_rate}"
    if momentum > 0:
        opt_info_str += f", momentum: {momentum}"
        if nesterov:
            opt_info_str += " (Nesterov)"
    if weight_decay > 0:
        opt_info_str += f", L2: {weight_decay}"

    optimizer_fn = create_sgd_optimizer(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov,
        weight_decay=weight_decay
    )

    return run_training(
        optimizer_name="SGD",
        optimizer_fn=optimizer_fn,
        batch_size=batch_size,
        num_epochs=num_epochs,
        hidden_size=hidden_size,
        data_root=data_root,
        device=device,
        use_module=use_sgd_module,
        verbose=verbose,
        opt_info_str=opt_info_str
    )


def run_adagrad_training(batch_size=64, num_epochs=10, learning_rate=0.01,
                          lr_decay=0.0, weight_decay=0.0, eps=1e-10,
                          hidden_size=128, data_root='./data', device=None,
                          use_adagrad_module=False, verbose=True):
    """
    Adagradを用いてMNISTデータを学習する関数

    Args:
        batch_size (int): バッチサイズ（デフォルト: 64）
        num_epochs (int): エポック数（デフォルト: 10）
        learning_rate (float): 学習率（デフォルト: 0.01）
        lr_decay (float): 学習率の減衰率（デフォルト: 0.0）
        weight_decay (float): 重み減衰係数（デフォルト: 0.0）
        eps (float): 数値安定性のための小さな値（デフォルト: 1e-10）
        hidden_size (int): 隠れ層のサイズ（デフォルト: 128）
        data_root (str): データを保存するディレクトリ（デフォルト: './data'）
        device (torch.device): 使用するデバイス（Noneの場合は自動選択）
        use_adagrad_module (bool): Adagrad.pyのtrain_with_adagrad関数を使用するかどうか
        verbose (bool): 進捗を表示するかどうか（デフォルト: True）

    Returns:
        tuple: (model, train_losses, test_losses, test_accuracy) のタプル
    """
    from Adagrad import create_adagrad_optimizer

    opt_info_str = f", lr: {learning_rate}"
    if lr_decay > 0:
        opt_info_str += f", lr_decay: {lr_decay}"

    optimizer_fn = create_adagrad_optimizer(
        learning_rate=learning_rate,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        eps=eps
    )

    return run_training(
        optimizer_name="Adagrad",
        optimizer_fn=optimizer_fn,
        batch_size=batch_size,
        num_epochs=num_epochs,
        hidden_size=hidden_size,
        data_root=data_root,
        device=device,
        use_module=use_adagrad_module,
        verbose=verbose,
        opt_info_str=opt_info_str
    )


def run_rmsprop_training(batch_size=64, num_epochs=10, learning_rate=0.01,
                          alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0,
                          centered=False, hidden_size=128, data_root='./data',
                          device=None, use_rmsprop_module=False, verbose=True):
    """
    RMSPropを用いてMNISTデータを学習する関数

    Args:
        batch_size (int): バッチサイズ（デフォルト: 64）
        num_epochs (int): エポック数（デフォルト: 10）
        learning_rate (float): 学習率（デフォルト: 0.01）
        alpha (float): 平滑化係数（デフォルト: 0.99）
        eps (float): 数値安定性のための小さな値（デフォルト: 1e-8）
        weight_decay (float): 重み減衰係数（デフォルト: 0.0）
        momentum (float): モーメンタム係数（デフォルト: 0.0）
        centered (bool): 中心化されたRMSPropを使用するかどうか（デフォルト: False）
        hidden_size (int): 隠れ層のサイズ（デフォルト: 128）
        data_root (str): データを保存するディレクトリ（デフォルト: './data'）
        device (torch.device): 使用するデバイス（Noneの場合は自動選択）
        use_rmsprop_module (bool): RMSProp.pyのtrain_with_rmsprop関数を使用するかどうか
        verbose (bool): 進捗を表示するかどうか（デフォルト: True）

    Returns:
        tuple: (model, train_losses, test_losses, test_accuracy) のタプル
    """
    from RMSProp import create_rmsprop_optimizer

    opt_info_str = f", lr: {learning_rate}, alpha: {alpha}"
    if momentum > 0:
        opt_info_str += f", momentum: {momentum}"
    if centered:
        opt_info_str += " (centered)"

    optimizer_fn = create_rmsprop_optimizer(
        learning_rate=learning_rate,
        alpha=alpha,
        eps=eps,
        weight_decay=weight_decay,
        momentum=momentum,
        centered=centered
    )

    return run_training(
        optimizer_name="RMSProp",
        optimizer_fn=optimizer_fn,
        batch_size=batch_size,
        num_epochs=num_epochs,
        hidden_size=hidden_size,
        data_root=data_root,
        device=device,
        use_module=use_rmsprop_module,
        verbose=verbose,
        opt_info_str=opt_info_str
    )


def run_adadelta_training(batch_size=64, num_epochs=10, learning_rate=1.0,
                          rho=0.9, eps=1e-6, weight_decay=0.0, hidden_size=128,
                          data_root='./data', device=None, use_adadelta_module=True,
                          verbose=True):
    """
    Adadeltaを用いてMNISTデータを学習する関数

    Args:
        batch_size (int): バッチサイズ（デフォルト: 64）
        num_epochs (int): エポック数（デフォルト: 10）
        learning_rate (float): 学習率（デフォルト: 1.0、Adadeltaでは通常1.0）
        rho (float): 減衰率（デフォルト: 0.9）
        eps (float): 数値安定性のための小さな値（デフォルト: 1e-6）
        weight_decay (float): 重み減衰係数（デフォルト: 0.0）
        hidden_size (int): 隠れ層のサイズ（デフォルト: 128）
        data_root (str): データを保存するディレクトリ（デフォルト: './data'）
        device (torch.device): 使用するデバイス（Noneの場合は自動選択）
        use_adadelta_module (bool): Adadelta.pyのtrain_with_adadelta関数を使用するかどうか
        verbose (bool): 進捗を表示するかどうか（デフォルト: True）

    Returns:
        tuple: (model, train_losses, test_losses, test_accuracy) のタプル
    """
    from Adadelta import create_adadelta_optimizer

    opt_info_str = f", lr: {learning_rate}, rho: {rho}"

    optimizer_fn = create_adadelta_optimizer(
        learning_rate=learning_rate,
        rho=rho,
        eps=eps,
        weight_decay=weight_decay
    )

    return run_training(
        optimizer_name="Adadelta",
        optimizer_fn=optimizer_fn,
        batch_size=batch_size,
        num_epochs=num_epochs,
        hidden_size=hidden_size,
        data_root=data_root,
        device=device,
        use_module=use_adadelta_module,
        verbose=verbose,
        opt_info_str=opt_info_str
    )


def run_cnn_training(batch_size=64, num_epochs=10, learning_rate=0.001,
                     model_type='simple', dropout_rate=0.25,
                     optimizer_type='adam', momentum=0.9,
                     data_root='./data', device=None, verbose=True,
                     save_models=True, model_save_dir='./models'):
    """
    CNNモデルを用いてMNISTデータを学習する関数

    Args:
        batch_size (int): バッチサイズ（デフォルト: 64）
        num_epochs (int): エポック数（デフォルト: 10）
        learning_rate (float): 学習率（デフォルト: 0.001）
        model_type (str): モデルタイプ ('simple' または 'deep')
        dropout_rate (float): ドロップアウト率（デフォルト: 0.25）
        optimizer_type (str): オプティマイザタイプ ('adam', 'sgd', 'rmsprop')
        momentum (float): SGDのモーメンタム係数（デフォルト: 0.9）
        data_root (str): データを保存するディレクトリ（デフォルト: './data'）
        device (torch.device): 使用するデバイス（Noneの場合は自動選択）
        verbose (bool): 進捗を表示するかどうか（デフォルト: True）
        save_models (bool): 各エポックでモデルを保存するかどうか（デフォルト: True）
        model_save_dir (str): モデル保存のベースディレクトリ（デフォルト: './models'）

    Returns:
        tuple: (model, train_losses, test_losses, test_accuracy, save_dir) のタプル
            - model: 学習済みモデル
            - train_losses: 各エポックの訓練損失のリスト
            - test_losses: 各エポックのテスト損失のリスト
            - test_accuracy: テストデータでの正解率
            - save_dir: モデル保存ディレクトリ（save_models=Falseの場合はNone）
    """
    print("=" * 50)
    print(f"MNIST Training Start (CNN - {model_type})")
    print("=" * 50)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # モデル保存ディレクトリの作成
    save_dir = None
    if save_models:
        save_dir = create_save_directory(model_save_dir)
        print(f"Model save directory: {save_dir}")

    # データの読み込み
    print("\nLoading data...")
    train_loader, test_loader = get_mnist_data_loaders(batch_size=batch_size, root=data_root)
    print(f"Training data: {len(train_loader.dataset)} samples")
    print(f"Test data: {len(test_loader.dataset)} samples")

    # モデルの作成
    print(f"\nCreating CNN model ({model_type})...")
    model = get_cnn_model(model_type=model_type, num_classes=10, dropout_rate=dropout_rate)
    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # オプティマイザの選択
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        opt_info = f"Adam (lr={learning_rate})"
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        opt_info = f"SGD (lr={learning_rate}, momentum={momentum})"
    elif optimizer_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        opt_info = f"RMSprop (lr={learning_rate})"
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    print(f"Optimizer: {opt_info}")

    # 損失関数
    criterion = nn.CrossEntropyLoss()

    # 学習
    train_losses = []
    test_losses = []
    best_accuracy = 0.0

    print(f"\nStarting training (epochs: {num_epochs})...")

    for epoch in range(num_epochs):
        # 訓練
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)

        # テストlossの計算
        avg_test_loss = calculate_test_loss(model, test_loader, criterion, device)
        test_losses.append(avg_test_loss)

        # エポックごとの精度を計算
        epoch_accuracy = evaluate_model(model, test_loader, device=device)

        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

        # モデルの保存
        if save_models and save_dir:
            save_epoch_model(
                model=model,
                save_dir=save_dir,
                epoch=epoch + 1,
                optimizer=optimizer,
                train_loss=avg_train_loss,
                test_loss=avg_test_loss,
                accuracy=epoch_accuracy
            )

            # ベストモデルの保存
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                save_best_model(model, save_dir, epoch + 1, epoch_accuracy, optimizer)

    # 最終評価
    print("\nEvaluating model...")
    test_accuracy = evaluate_model(model, test_loader, device=device)
    print(f"Test accuracy: {test_accuracy:.2f}%")

    # Confusion Matrixの表示
    plot_confusion_matrix(model, test_loader, device=device)

    print("\n" + "=" * 50)
    print("Training Complete")
    if save_dir:
        print(f"Models saved to: {save_dir}")
    print("=" * 50)

    return model, train_losses, test_losses, test_accuracy, save_dir


def predict_single_image(model, image, device=None):
    """
    単一の画像に対して予測を行う

    Args:
        model (nn.Module): 学習済みモデル
        image (torch.Tensor): 入力画像 (1, 28, 28) or (28, 28)
        device (torch.device): 使用するデバイス

    Returns:
        tuple: (predicted_class, probabilities)
            - predicted_class: 予測クラス（0-9）
            - probabilities: 各クラスの確率
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    # 入力の形状を調整
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # (28, 28) -> (1, 1, 28, 28)
    elif image.dim() == 3:
        image = image.unsqueeze(0)  # (1, 28, 28) -> (1, 1, 28, 28)

    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities.squeeze().cpu().numpy()


def predict_batch(model, images, device=None):
    """
    バッチの画像に対して予測を行う

    Args:
        model (nn.Module): 学習済みモデル
        images (torch.Tensor): 入力画像バッチ (N, 1, 28, 28)
        device (torch.device): 使用するデバイス

    Returns:
        tuple: (predicted_classes, probabilities)
            - predicted_classes: 予測クラスのリスト
            - probabilities: 各クラスの確率のリスト
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    images = images.to(device)

    with torch.no_grad():
        output = model(images)
        probabilities = torch.softmax(output, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)

    return predicted_classes.cpu().numpy(), probabilities.cpu().numpy()


def run_inference_from_saved_model(model_path, model_type='simple', dropout_rate=0.25,
                                   data_root='./data', batch_size=64, device=None):
    """
    保存されたモデルを読み込んでテストデータに対して推論を行う

    Args:
        model_path (str): 保存されたモデルのパス
        model_type (str): モデルタイプ ('simple' または 'deep')
        dropout_rate (float): ドロップアウト率
        data_root (str): データを保存するディレクトリ
        batch_size (int): バッチサイズ
        device (torch.device): 使用するデバイス

    Returns:
        tuple: (model, accuracy, predictions, targets)
    """
    print("=" * 50)
    print("Running Inference from Saved Model")
    print("=" * 50)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # モデルの作成と読み込み
    print(f"\nLoading model from: {model_path}")
    model = get_cnn_model(model_type=model_type, num_classes=10, dropout_rate=dropout_rate)
    checkpoint = load_model(model, model_path, device)

    # テストデータの読み込み
    print("\nLoading test data...")
    _, test_loader = get_mnist_data_loaders(batch_size=batch_size, root=data_root)
    print(f"Test data: {len(test_loader.dataset)} samples")

    # 推論の実行
    print("\nRunning inference...")
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            predictions = torch.argmax(output, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.numpy())

    # 精度の計算
    correct = sum(p == t for p, t in zip(all_predictions, all_targets))
    accuracy = 100 * correct / len(all_targets)

    print(f"\nInference Results:")
    print(f"  Total samples: {len(all_targets)}")
    print(f"  Correct predictions: {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")

    # Confusion Matrixの表示
    print_confusion_matrix(model, test_loader, device=device)

    print("\n" + "=" * 50)
    print("Inference Complete")
    print("=" * 50)

    return model, accuracy, all_predictions, all_targets


def interactive_inference(model_path=None, model=None, model_type='simple',
                         dropout_rate=0.25, data_root='./data', device=None):
    """
    対話的に推論を実行する（テストデータからランダムにサンプルを選択）

    Args:
        model_path (str): 保存されたモデルのパス（modelがNoneの場合に使用）
        model (nn.Module): 学習済みモデル（指定された場合はmodel_pathを無視）
        model_type (str): モデルタイプ
        dropout_rate (float): ドロップアウト率
        data_root (str): データを保存するディレクトリ
        device (torch.device): 使用するデバイス

    Returns:
        nn.Module: 使用したモデル
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # モデルの準備
    if model is None:
        if model_path is None:
            # 最新のモデルディレクトリからbest_modelを読み込む
            latest_dir = get_latest_model_dir()
            if latest_dir is None:
                print("No saved models found.")
                return None
            model_path = f"{latest_dir}/best_model.pth"
            print(f"Using latest model: {model_path}")

        model = get_cnn_model(model_type=model_type, num_classes=10, dropout_rate=dropout_rate)
        load_model(model, model_path, device)

    model = model.to(device)
    model.eval()

    # テストデータの読み込み
    _, test_loader = get_mnist_data_loaders(batch_size=1, root=data_root)
    test_dataset = test_loader.dataset

    print("\n" + "=" * 50)
    print("Interactive Inference Mode")
    print("=" * 50)
    print("Randomly selecting samples from test data...")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            # ランダムにサンプルを選択
            idx = np.random.randint(0, len(test_dataset))
            image, true_label = test_dataset[idx]

            # 予測
            predicted_class, probs = predict_single_image(model, image, device)

            print(f"Sample #{idx}")
            print(f"  True label: {true_label}")
            print(f"  Predicted:  {predicted_class}")
            print(f"  Confidence: {probs[predicted_class]*100:.2f}%")
            print(f"  Correct:    {'Yes' if predicted_class == true_label else 'No'}")
            print("-" * 30)

            input("Press Enter for next sample (Ctrl+C to exit)...")

    except KeyboardInterrupt:
        print("\nExiting interactive mode.")

    return model


def parse_arguments():
    """
    Parse command line arguments
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='MNIST CNN Training and Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py --train                    # Train from scratch
  python main.py --train --epochs 10        # Train for 10 epochs
  python main.py --inference                # Run inference with latest model
  python main.py --inference --model path   # Run inference with specific model
  python main.py --interactive              # Interactive inference mode
  python main.py --list                     # List saved models
        '''
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--train', action='store_true',
                           help='Train a new model from scratch')
    mode_group.add_argument('--inference', action='store_true',
                           help='Run inference using a saved model')
    mode_group.add_argument('--interactive', action='store_true',
                           help='Interactive inference mode')
    mode_group.add_argument('--list', action='store_true',
                           help='List saved models')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs for training (default: 5)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--model-type', type=str, default='simple',
                       choices=['simple', 'deep'],
                       help='Model type: simple or deep (default: simple)')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'rmsprop'],
                       help='Optimizer type (default: adam)')
    parser.add_argument('--dropout', type=float, default=0.25,
                       help='Dropout rate (default: 0.25)')

    # Model save/load parameters
    parser.add_argument('--save-dir', type=str, default='./models',
                       help='Directory to save models (default: ./models)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file for inference (default: latest best model)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save models during training')

    # Other parameters
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory for MNIST data (default: ./data)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')

    return parser.parse_args()


def main():
    """
    Main function
    """
    args = parse_arguments()

    # Determine mode
    if args.train:
        # Training mode
        print("=" * 60)
        print("MODE: Training from scratch")
        print("=" * 60)

        model, train_losses, test_losses, accuracy, save_dir = run_cnn_training(
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            model_type=args.model_type,
            dropout_rate=args.dropout,
            optimizer_type=args.optimizer,
            data_root=args.data_dir,
            verbose=not args.quiet,
            save_models=not args.no_save,
            model_save_dir=args.save_dir
        )

        print(f"\nFinal test accuracy: {accuracy:.2f}%")
        print(f"Final train loss: {train_losses[-1]:.4f}")
        print(f"Final test loss: {test_losses[-1]:.4f}")
        if save_dir:
            print(f"\nModels saved to: {save_dir}")

    elif args.inference:
        # Inference mode
        print("=" * 60)
        print("MODE: Inference")
        print("=" * 60)

        if args.model:
            model_path = args.model
        else:
            latest_dir = get_latest_model_dir(args.save_dir)
            if latest_dir:
                model_path = f"{latest_dir}/best_model.pth"
            else:
                print("No saved models found. Please train a model first.")
                print("  python main.py --train")
                return

        run_inference_from_saved_model(
            model_path=model_path,
            model_type=args.model_type,
            dropout_rate=args.dropout,
            data_root=args.data_dir,
            batch_size=args.batch_size
        )

    elif args.interactive:
        # Interactive inference mode
        print("=" * 60)
        print("MODE: Interactive Inference")
        print("=" * 60)

        model_path = args.model
        if model_path is None:
            latest_dir = get_latest_model_dir(args.save_dir)
            if latest_dir:
                model_path = f"{latest_dir}/best_model.pth"

        interactive_inference(
            model_path=model_path,
            model_type=args.model_type,
            dropout_rate=args.dropout,
            data_root=args.data_dir
        )

    elif args.list:
        # List saved models
        print("=" * 60)
        print("Saved Models")
        print("=" * 60)

        import os
        if not os.path.exists(args.save_dir):
            print(f"No model directory found: {args.save_dir}")
            return

        dirs = []
        for name in sorted(os.listdir(args.save_dir)):
            path = os.path.join(args.save_dir, name)
            if os.path.isdir(path):
                dirs.append(path)

        if not dirs:
            print("No saved models found.")
            return

        for d in dirs:
            print(f"\n{d}/")
            models = list_saved_models(d)
            for m in models:
                # Load checkpoint to show info
                try:
                    checkpoint = torch.load(m, map_location='cpu')
                    info = []
                    if 'epoch' in checkpoint:
                        info.append(f"epoch={checkpoint['epoch']}")
                    if 'accuracy' in checkpoint:
                        info.append(f"acc={checkpoint['accuracy']:.2f}%")
                    if 'train_loss' in checkpoint:
                        info.append(f"loss={checkpoint['train_loss']:.4f}")
                    info_str = ", ".join(info) if info else ""
                    print(f"  - {os.path.basename(m)}: {info_str}")
                except:
                    print(f"  - {os.path.basename(m)}")

    else:
        # Default: Show help and run training
        print("=" * 60)
        print("MNIST CNN Training and Inference")
        print("=" * 60)
        print("\nNo mode specified. Running default training...")
        print("Use --help to see all options.\n")

        model, train_losses, test_losses, accuracy, save_dir = run_cnn_training(
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            model_type=args.model_type,
            dropout_rate=args.dropout,
            optimizer_type=args.optimizer,
            data_root=args.data_dir,
            verbose=not args.quiet,
            save_models=not args.no_save,
            model_save_dir=args.save_dir
        )

        print(f"\nFinal test accuracy: {accuracy:.2f}%")
        print(f"Final train loss: {train_losses[-1]:.4f}")
        print(f"Final test loss: {test_losses[-1]:.4f}")
        if save_dir:
            print(f"\nModels saved to: {save_dir}")
            print("\nNext steps:")
            print("  python main.py --inference      # Run inference")
            print("  python main.py --interactive    # Interactive mode")
            print("  python main.py --list           # List saved models")


if __name__ == "__main__":
    main()

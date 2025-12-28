"""
MNISTデータの学習と評価を実行するメインファイル
"""
import torch
import torch.nn as nn
import numpy as np
from utils import get_mnist_data_loaders
from SGD import train_with_sgd
try:
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("警告: sklearn, matplotlib, seabornがインストールされていません。confusion matrixの可視化はスキップされます。")


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
        print("可視化ライブラリがインストールされていないため、confusion matrixを表示できません。")
        return
    
    cm, _, _ = calculate_confusion_matrix(model, test_loader, device)
    
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'サンプル数'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('実際のラベル', fontsize=12)
    plt.xlabel('予測ラベル', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrixを {save_path} に保存しました。")
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
    print("\n実際のラベル（行）と予測ラベル（列）の対応:")
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
    print("\n各クラスの精度:")
    for i in range(10):
        correct = cm[i, i]
        total = cm[i, :].sum()
        accuracy = 100 * correct / total if total > 0 else 0
        print(f"  クラス {i}: {correct}/{total} = {accuracy:.2f}%")
    
    print("=" * 50)


def train_model(model, train_loader, test_loader, optimizer_fn, num_epochs=10,
                device=None, verbose=True):
    """
    モデルを学習する汎用的な関数
    
    Args:
        model (nn.Module): 学習するモデル
        train_loader (DataLoader): 訓練データのDataLoader
        test_loader (DataLoader): テストデータのDataLoader
        optimizer_fn (callable): オプティマイザーを返す関数（model.parameters()を引数に取る）
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
        optimizer_fn (callable): オプティマイザーを作成する関数
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
    print(f"MNIST学習開始 ({optimizer_name})")
    print("=" * 50)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    print("\nデータを読み込んでいます...")
    train_loader, test_loader = get_mnist_data_loaders(batch_size=batch_size, root=data_root)
    print(f"訓練データ: {len(train_loader.dataset)}サンプル")
    print(f"テストデータ: {len(test_loader.dataset)}サンプル")
    
    print(f"\nモデルを作成しています...")
    model = SimpleMNISTModel(input_size=28*28, hidden_size=hidden_size, num_classes=10)
    print(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    opt_info = f"エポック数: {num_epochs}" + opt_info_str
    print(f"\n学習を開始します（{opt_info}）...")
    
    if use_module:
        # 専用モジュールのtrain関数を使用する場合
        # この場合は各最適化手法のモジュールからtrain関数をインポートして使用
        # ただし、現在の実装ではtrain_modelを使用する方が簡潔
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
    
    print("\nモデルを評価しています...")
    test_accuracy = evaluate_model(model, test_loader, device=device)
    print(f"テストデータでの正解率: {test_accuracy:.2f}%")
    
    # Confusion Matrixの表示
    print_confusion_matrix(model, test_loader, device=device)
    
    print("\n" + "=" * 50)
    print("学習完了")
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
        nesterov (bool): ネステロフの加速法を使用するかどうか（デフォルト: False）
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
    
    opt_info_str = f", 学習率: {learning_rate}"
    if momentum > 0:
        opt_info_str += f", モーメンタム: {momentum}"
        if nesterov:
            opt_info_str += " (Nesterov)"
    if weight_decay > 0:
        opt_info_str += f", L2正則化: {weight_decay}"
    
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
    
    opt_info_str = f", 学習率: {learning_rate}"
    if lr_decay > 0:
        opt_info_str += f", 学習率減衰: {lr_decay}"
    
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
        alpha (float): 平滑化定数（デフォルト: 0.99）
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
    
    opt_info_str = f", 学習率: {learning_rate}, alpha: {alpha}"
    if momentum > 0:
        opt_info_str += f", モーメンタム: {momentum}"
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
    
    opt_info_str = f", 学習率: {learning_rate}, rho: {rho}"
    
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


if __name__ == "__main__":
    # SGDの例
    # model, train_losses, test_losses, accuracy = run_sgd_training(
    #     batch_size=64,
    #     num_epochs=5,
    #     learning_rate=0.01,
    #     momentum=0.0,
    #     nesterov=False,
    #     hidden_size=128
    # )
    
    # Adagradの例
    # model, train_losses, test_losses, accuracy = run_adagrad_training(
    #     batch_size=64,
    #     num_epochs=5,
    #     learning_rate=0.01,
    #     hidden_size=128
    # )
    
    # RMSPropの例
    # model, train_losses, test_losses, accuracy = run_rmsprop_training(
    #     batch_size=64,
    #     num_epochs=5,
    #     learning_rate=0.001,
    #     hidden_size=128
    # )
    
    # Adadeltaの例
    model, train_losses, test_losses, accuracy = run_adadelta_training(
        batch_size=64,
        num_epochs=5,
        learning_rate=1.0,
        hidden_size=128
    )
    
    print(f"\n最終テスト正解率: {accuracy:.2f}%")
    print(f"最終訓練損失: {train_losses[-1]:.4f}")
    print(f"最終テスト損失: {test_losses[-1]:.4f}")


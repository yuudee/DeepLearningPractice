# -*- coding: utf-8 -*-
"""
IMDb感情分析 - メインスクリプト

使用方法:
    # 学習
    python main.py --train --model lstm --epochs 10

    # 推論（最新のモデルを使用）
    python main.py --inference

    # 特定のモデルで推論
    python main.py --inference --model-path ./models/20240101_120000/best_model.pth

    # 対話的推論
    python main.py --interactive
"""
import argparse
import os
import sys
import torch
import torch.nn as nn

from models import get_model, count_parameters
from data_utils import load_imdb_data, create_data_loaders, preprocess_text
from trainer import train_model, evaluate, predict_sentiment
from model_utils import (
    get_save_dir, create_save_callback, save_training_config,
    load_model_for_inference, get_latest_model_dir, list_saved_models,
    load_checkpoint, load_vocab
)


def parse_args():
    """
    コマンドライン引数をパース
    """
    parser = argparse.ArgumentParser(
        description='IMDb Sentiment Analysis with RNN/LSTM/GRU/UGRNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Train a new model
  python main.py --train --model lstm --epochs 10

  # Train with specific parameters
  python main.py --train --model gru --epochs 20 --lr 0.0005 --hidden-dim 512

  # Run inference with latest model
  python main.py --inference

  # Run inference with specific model
  python main.py --inference --model-path ./models/20240101_120000/best_model.pth

  # Interactive inference
  python main.py --interactive

  # List saved models
  python main.py --list
        '''
    )

    # モード選択
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--train', action='store_true',
                           help='Train a new model')
    mode_group.add_argument('--inference', action='store_true',
                           help='Run inference on test set')
    mode_group.add_argument('--interactive', action='store_true',
                           help='Interactive inference mode')
    mode_group.add_argument('--list', action='store_true',
                           help='List saved models')

    # モデル設定
    parser.add_argument('--model', type=str, default='lstm',
                       choices=['rnn', 'lstm', 'gru', 'ugrnn'],
                       help='Model type (default: lstm)')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimension (default: 128)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension (default: 256)')
    parser.add_argument('--n-layers', type=int, default=2,
                       help='Number of RNN layers (default: 2)')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                       help='Use bidirectional RNN (default: True)')
    parser.add_argument('--no-bidirectional', action='store_false', dest='bidirectional',
                       help='Use unidirectional RNN')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate (default: 0.5)')

    # データ設定
    parser.add_argument('--vocab-size', type=int, default=25000,
                       help='Vocabulary size (default: 25000)')
    parser.add_argument('--max-length', type=int, default=256,
                       help='Maximum sequence length (default: 256)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory (default: ./data)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples per class (default: all)')

    # 学習設定
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'rmsprop'],
                       help='Optimizer (default: adam)')

    # モデル保存・読み込み
    parser.add_argument('--save-dir', type=str, default='./models',
                       help='Model save directory (default: ./models)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model for inference')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save models')

    # その他
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    parser.add_argument('--no-lengths', action='store_true',
                       help='Do not use sequence lengths (for debugging)')

    return parser.parse_args()


def get_optimizer(model, optimizer_type, lr):
    """
    オプティマイザを取得
    """
    if optimizer_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_type == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def run_training(args):
    """
    学習を実行
    """
    print("=" * 60)
    print("MODE: Training")
    print("=" * 60)

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # データの読み込み
    print("\nLoading data...")
    train_texts, train_labels, test_texts, test_labels = load_imdb_data(
        data_dir=args.data_dir,
        max_samples=args.max_samples
    )

    # データローダーの作成
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader, vocab = create_data_loaders(
        train_texts, train_labels, test_texts, test_labels,
        vocab_size=args.vocab_size,
        max_length=args.max_length,
        batch_size=args.batch_size
    )

    # モデルの作成
    print(f"\nCreating {args.model.upper()} model...")
    model_config = {
        'vocab_size': len(vocab),
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'output_dim': 1,
        'n_layers': args.n_layers,
        'bidirectional': args.bidirectional,
        'dropout': args.dropout,
        'pad_idx': vocab.pad_idx
    }

    model = get_model(args.model, **model_config)
    model = model.to(device)

    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")

    # オプティマイザと損失関数
    optimizer = get_optimizer(model, args.optimizer, args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # 保存ディレクトリの作成
    save_dir = None
    save_callback = None
    if not args.no_save:
        save_dir = get_save_dir(args.save_dir)
        print(f"\nModels will be saved to: {save_dir}")

        # 学習設定を保存
        training_config = {
            'model_type': args.model,
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'n_layers': args.n_layers,
            'bidirectional': args.bidirectional,
            'dropout': args.dropout,
            'vocab_size': args.vocab_size,
            'max_length': args.max_length,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'optimizer': args.optimizer
        }
        save_training_config(save_dir, training_config)

        # 保存コールバックの作成
        # model_configを渡してモデル再構築に必要な情報を保存
        save_model_config = {
            'model_type': args.model,
            **model_config
        }
        save_callback = create_save_callback(
            save_dir=save_dir,
            vocab=vocab,
            model_config=save_model_config,
            optimizer=optimizer,
            verbose=not args.quiet
        )

    # 学習
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=args.epochs,
        save_callback=save_callback,
        verbose=not args.quiet,
        use_lengths=not args.no_lengths
    )

    # テストセットでの評価
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device,
        use_lengths=not args.no_lengths
    )
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")

    if save_dir:
        print(f"\nModels saved to: {save_dir}")
        print("\nTo run inference:")
        print(f"  python main.py --inference --model-path {save_dir}/best_model.pth")

    return model, history


def run_inference(args):
    """
    推論を実行
    """
    print("=" * 60)
    print("MODE: Inference")
    print("=" * 60)

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # モデルパスの決定
    if args.model_path:
        model_path = args.model_path
    else:
        latest_dir = get_latest_model_dir(args.save_dir)
        if latest_dir is None:
            print("No saved models found. Please train a model first.")
            print("  python main.py --train")
            return
        model_path = os.path.join(latest_dir, 'best_model.pth')

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"\nLoading model from: {model_path}")

    # チェックポイントの読み込み
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']

    # model_typeをconfigから取り出してget_modelに渡す
    model_type = config.pop('model_type', args.model)

    # モデルの作成
    model = get_model(model_type, **config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model type: {model_type.upper()}")
    print(f"Loaded from epoch: {checkpoint['epoch']}")
    print(f"Validation accuracy at save: {checkpoint['val_acc']*100:.2f}%")

    # 語彙の読み込み
    vocab_path = os.path.join(os.path.dirname(model_path), 'vocab.pkl')
    vocab = load_vocab(vocab_path)

    # テストデータの読み込み
    print("\nLoading test data...")
    _, _, test_texts, test_labels = load_imdb_data(
        data_dir=args.data_dir,
        max_samples=args.max_samples
    )

    # データローダーの作成（テストデータのみ）
    from data_utils import IMDbDataset
    from torch.utils.data import DataLoader

    test_dataset = IMDbDataset(test_texts, test_labels, vocab, args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 評価
    print("\nEvaluating...")
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device,
        use_lengths=not args.no_lengths
    )
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")


def run_interactive(args):
    """
    対話的推論モード
    """
    print("=" * 60)
    print("MODE: Interactive Inference")
    print("=" * 60)

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # モデルパスの決定
    if args.model_path:
        model_path = args.model_path
    else:
        latest_dir = get_latest_model_dir(args.save_dir)
        if latest_dir is None:
            print("No saved models found. Please train a model first.")
            return
        model_path = os.path.join(latest_dir, 'best_model.pth')

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"\nLoading model from: {model_path}")

    # チェックポイントの読み込み
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']
    model_type = config.pop('model_type', args.model)

    model = get_model(model_type, **config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 語彙の読み込み
    vocab_path = os.path.join(os.path.dirname(model_path), 'vocab.pkl')
    vocab = load_vocab(vocab_path)

    print(f"\nModel loaded: {model_type.upper()}")
    print("Enter a movie review to analyze sentiment.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            text = input("Review: ").strip()

            if not text:
                continue

            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            # 予測
            sentiment, probability = predict_sentiment(
                model, text, vocab, args.max_length, device,
                use_lengths=not args.no_lengths
            )

            print(f"Sentiment: {sentiment.upper()}")
            print(f"Confidence: {probability*100:.1f}% positive, {(1-probability)*100:.1f}% negative\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def run_list_models(args):
    """
    保存されたモデルの一覧を表示
    """
    print("=" * 60)
    print("Saved Models")
    print("=" * 60)

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
            try:
                checkpoint = torch.load(m, map_location='cpu')
                info = []
                if 'epoch' in checkpoint:
                    info.append(f"epoch={checkpoint['epoch']}")
                if 'val_acc' in checkpoint:
                    info.append(f"val_acc={checkpoint['val_acc']*100:.2f}%")
                if 'model_config' in checkpoint and 'model_type' in checkpoint['model_config']:
                    info.append(f"type={checkpoint['model_config']['model_type']}")
                info_str = ", ".join(info) if info else ""
                print(f"  - {os.path.basename(m)}: {info_str}")
            except:
                print(f"  - {os.path.basename(m)}")


def main():
    """
    メイン関数
    """
    args = parse_args()

    if args.train:
        run_training(args)
    elif args.inference:
        run_inference(args)
    elif args.interactive:
        run_interactive(args)
    elif args.list:
        run_list_models(args)
    else:
        # デフォルトはヘルプを表示して学習
        print("No mode specified. Use --help for options.\n")
        print("Running default training with LSTM model...\n")
        args.train = True
        run_training(args)


if __name__ == "__main__":
    main()


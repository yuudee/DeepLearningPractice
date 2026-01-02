# -*- coding: utf-8 -*-
"""
IMDbデータセットの読み込みと前処理
"""
import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import urllib.request
import tarfile
import random


class Vocabulary:
    """
    単語から整数へのマッピングを管理するクラス
    """
    def __init__(self, max_size=None, min_freq=1):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()

        # 特殊トークン
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.pad_idx = 0
        self.unk_idx = 1

    def build_vocab(self, texts):
        """
        テキストから語彙を構築
        """
        for text in texts:
            self.word_freq.update(text.split())

        # 特殊トークンを追加
        self.word2idx[self.pad_token] = self.pad_idx
        self.word2idx[self.unk_token] = self.unk_idx
        self.idx2word[self.pad_idx] = self.pad_token
        self.idx2word[self.unk_idx] = self.unk_token

        # 頻度でフィルタリング
        words = [word for word, freq in self.word_freq.most_common()
                 if freq >= self.min_freq]

        # 最大サイズでカット
        if self.max_size is not None:
            words = words[:self.max_size - 2]  # 特殊トークン分を引く

        for idx, word in enumerate(words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def __len__(self):
        return len(self.word2idx)

    def encode(self, text):
        """
        テキストを整数のリストに変換
        """
        return [self.word2idx.get(word, self.unk_idx) for word in text.split()]

    def decode(self, indices):
        """
        整数のリストをテキストに変換
        """
        return ' '.join([self.idx2word.get(idx, self.unk_token) for idx in indices])


class IMDbDataset(Dataset):
    """
    IMDbデータセット
    """
    def __init__(self, texts, labels, vocab, max_length=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # エンコード
        encoded = self.vocab.encode(text)

        # 長さの調整
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        length = len(encoded)

        # パディング
        if len(encoded) < self.max_length:
            encoded = encoded + [self.vocab.pad_idx] * (self.max_length - len(encoded))

        return {
            'text': torch.tensor(encoded, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float),
            'length': torch.tensor(length, dtype=torch.long)
        }


def preprocess_text(text):
    """
    テキストの前処理
    """
    # HTMLタグの除去
    text = re.sub(r'<[^>]+>', '', text)
    # 小文字化
    text = text.lower()
    # 不要な文字の除去（アルファベットとスペースのみ残す）
    text = re.sub(r'[^a-z\s]', ' ', text)
    # 連続するスペースを1つに
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def download_imdb(data_dir='./data'):
    """
    IMDbデータセットをダウンロード
    """
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    tar_path = os.path.join(data_dir, 'aclImdb_v1.tar.gz')
    extract_path = os.path.join(data_dir, 'aclImdb')

    if os.path.exists(extract_path):
        print("IMDb dataset already exists.")
        return extract_path

    os.makedirs(data_dir, exist_ok=True)

    print("Downloading IMDb dataset...")
    urllib.request.urlretrieve(url, tar_path)

    print("Extracting...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(data_dir)

    # 圧縮ファイルを削除
    os.remove(tar_path)

    print("Done!")
    return extract_path


def load_imdb_data(data_dir='./data', max_samples=None):
    """
    IMDbデータを読み込む

    Args:
        data_dir (str): データディレクトリ
        max_samples (int): 最大サンプル数（Noneの場合は全て）

    Returns:
        train_texts, train_labels, test_texts, test_labels
    """
    # ダウンロード
    imdb_dir = download_imdb(data_dir)

    def load_split(split):
        texts = []
        labels = []

        for label_type, label_value in [('pos', 1), ('neg', 0)]:
            label_dir = os.path.join(imdb_dir, split, label_type)
            files = os.listdir(label_dir)

            if max_samples is not None:
                files = files[:max_samples // 2]

            for fname in files:
                fpath = os.path.join(label_dir, fname)
                with open(fpath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    text = preprocess_text(text)
                    texts.append(text)
                    labels.append(label_value)

        return texts, labels

    print("Loading training data...")
    train_texts, train_labels = load_split('train')

    print("Loading test data...")
    test_texts, test_labels = load_split('test')

    # シャッフル
    train_data = list(zip(train_texts, train_labels))
    test_data = list(zip(test_texts, test_labels))
    random.shuffle(train_data)
    random.shuffle(test_data)
    train_texts, train_labels = zip(*train_data)
    test_texts, test_labels = zip(*test_data)

    return list(train_texts), list(train_labels), list(test_texts), list(test_labels)


def create_data_loaders(train_texts, train_labels, test_texts, test_labels,
                        vocab_size=25000, min_freq=5, max_length=256,
                        batch_size=64, val_split=0.1):
    """
    データローダーを作成

    Args:
        train_texts: 学習用テキスト
        train_labels: 学習用ラベル
        test_texts: テスト用テキスト
        test_labels: テスト用ラベル
        vocab_size: 語彙サイズ
        min_freq: 最低出現頻度
        max_length: 最大シーケンス長
        batch_size: バッチサイズ
        val_split: 検証データの割合

    Returns:
        train_loader, val_loader, test_loader, vocab
    """
    # 語彙の構築
    print("Building vocabulary...")
    vocab = Vocabulary(max_size=vocab_size, min_freq=min_freq)
    vocab.build_vocab(train_texts)
    print(f"Vocabulary size: {len(vocab)}")

    # 検証データの分割
    val_size = int(len(train_texts) * val_split)
    val_texts = train_texts[:val_size]
    val_labels = train_labels[:val_size]
    train_texts = train_texts[val_size:]
    train_labels = train_labels[val_size:]

    # データセットの作成
    train_dataset = IMDbDataset(train_texts, train_labels, vocab, max_length)
    val_dataset = IMDbDataset(val_texts, val_labels, vocab, max_length)
    test_dataset = IMDbDataset(test_texts, test_labels, vocab, max_length)

    # データローダーの作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, vocab


def get_sample_batch(data_loader):
    """
    データローダーからサンプルバッチを取得
    """
    for batch in data_loader:
        return batch
    return None


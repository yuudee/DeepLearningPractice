# -*- coding: utf-8 -*-
"""
IMDb感情分析のためのRNN系モデル定義
- RNN (Vanilla RNN)
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- UGRNN (Update Gate RNN)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UGRNNCell(nn.Module):
    """
    Update Gate RNN (UGRNN) セル
    GRUの簡略化版で、リセットゲートを持たない

    h_t = (1 - z_t) * h_{t-1} + z_t * tanh(W_h * x_t + U_h * h_{t-1} + b_h)
    z_t = sigmoid(W_z * x_t + U_z * h_{t-1} + b_z)
    """
    def __init__(self, input_size, hidden_size):
        super(UGRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Update gate weights
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)

        # Candidate hidden state weights
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h_prev):
        """
        Args:
            x: (batch_size, input_size)
            h_prev: (batch_size, hidden_size)
        Returns:
            h_new: (batch_size, hidden_size)
        """
        # Update gate
        z = torch.sigmoid(self.W_z(x) + self.U_z(h_prev))

        # Candidate hidden state
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(h_prev))

        # New hidden state
        h_new = (1 - z) * h_prev + z * h_tilde

        return h_new


class UGRNN(nn.Module):
    """
    Update Gate RNN レイヤー
    """
    def __init__(self, input_size, hidden_size, num_layers=1,
                 batch_first=True, dropout=0.0, bidirectional=False):
        super(UGRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Create UGRNN cells for each layer and direction
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            for direction in range(self.num_directions):
                self.cells.append(UGRNNCell(layer_input_size, hidden_size))

        self.dropout = nn.Dropout(dropout) if dropout > 0 and num_layers > 1 else None

    def forward(self, x, h_0=None):
        """
        Args:
            x: (batch_size, seq_len, input_size) if batch_first else (seq_len, batch_size, input_size)
            h_0: (num_layers * num_directions, batch_size, hidden_size)
        Returns:
            output: (batch_size, seq_len, hidden_size * num_directions)
            h_n: (num_layers * num_directions, batch_size, hidden_size)
        """
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
        else:
            seq_len, batch_size, _ = x.size()
            x = x.transpose(0, 1)

        if h_0 is None:
            h_0 = torch.zeros(self.num_layers * self.num_directions,
                             batch_size, self.hidden_size, device=x.device)

        h_n_list = []
        layer_input = x

        for layer in range(self.num_layers):
            # Forward direction
            h = h_0[layer * self.num_directions]
            forward_outputs = []
            for t in range(seq_len):
                h = self.cells[layer * self.num_directions](layer_input[:, t, :], h)
                forward_outputs.append(h)
            forward_output = torch.stack(forward_outputs, dim=1)
            h_n_list.append(h)

            if self.bidirectional:
                # Backward direction
                h = h_0[layer * self.num_directions + 1]
                backward_outputs = []
                for t in range(seq_len - 1, -1, -1):
                    h = self.cells[layer * self.num_directions + 1](layer_input[:, t, :], h)
                    backward_outputs.append(h)
                backward_output = torch.stack(backward_outputs[::-1], dim=1)
                h_n_list.append(h)

                layer_output = torch.cat([forward_output, backward_output], dim=2)
            else:
                layer_output = forward_output

            if self.dropout is not None and layer < self.num_layers - 1:
                layer_output = self.dropout(layer_output)

            layer_input = layer_output

        h_n = torch.stack(h_n_list, dim=0)

        if not self.batch_first:
            layer_output = layer_output.transpose(0, 1)

        return layer_output, h_n


class SentimentRNN(nn.Module):
    """
    Vanilla RNNを使用した感情分析モデル
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 output_dim=1, n_layers=2, bidirectional=True,
                 dropout=0.5, pad_idx=0):
        super(SentimentRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers,
                         bidirectional=bidirectional, dropout=dropout if n_layers > 1 else 0,
                         batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths=None):
        # text: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(text))

        if text_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, hidden = self.rnn(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, hidden = self.rnn(embedded)

        # hidden: (n_layers * num_directions, batch_size, hidden_dim)
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        hidden = self.dropout(hidden)
        return self.fc(hidden)


class SentimentLSTM(nn.Module):
    """
    LSTMを使用した感情分析モデル
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 output_dim=1, n_layers=2, bidirectional=True,
                 dropout=0.5, pad_idx=0):
        super(SentimentLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                           bidirectional=bidirectional, dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths=None):
        embedded = self.dropout(self.embedding(text))

        if text_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, (hidden, cell) = self.lstm(embedded)

        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        hidden = self.dropout(hidden)
        return self.fc(hidden)


class SentimentGRU(nn.Module):
    """
    GRUを使用した感情分析モデル
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 output_dim=1, n_layers=2, bidirectional=True,
                 dropout=0.5, pad_idx=0):
        super(SentimentGRU, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers,
                         bidirectional=bidirectional, dropout=dropout if n_layers > 1 else 0,
                         batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths=None):
        embedded = self.dropout(self.embedding(text))

        if text_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, hidden = self.gru(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, hidden = self.gru(embedded)

        if self.gru.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        hidden = self.dropout(hidden)
        return self.fc(hidden)


class SentimentUGRNN(nn.Module):
    """
    UGRNNを使用した感情分析モデル
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 output_dim=1, n_layers=2, bidirectional=True,
                 dropout=0.5, pad_idx=0):
        super(SentimentUGRNN, self).__init__()

        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.ugrnn = UGRNN(embedding_dim, hidden_dim, num_layers=n_layers,
                          bidirectional=bidirectional, dropout=dropout if n_layers > 1 else 0,
                          batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths=None):
        embedded = self.dropout(self.embedding(text))

        output, hidden = self.ugrnn(embedded)

        if self.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        hidden = self.dropout(hidden)
        return self.fc(hidden)


def get_model(model_type, vocab_size, embedding_dim=128, hidden_dim=256,
              output_dim=1, n_layers=2, bidirectional=True, dropout=0.5, pad_idx=0):
    """
    モデルタイプに応じたモデルを返すファクトリ関数

    Args:
        model_type (str): モデルタイプ ('rnn', 'lstm', 'gru', 'ugrnn')
        vocab_size (int): 語彙サイズ
        embedding_dim (int): 埋め込み次元
        hidden_dim (int): 隠れ層の次元
        output_dim (int): 出力次元
        n_layers (int): RNN層の数
        bidirectional (bool): 双方向RNNを使用するか
        dropout (float): ドロップアウト率
        pad_idx (int): パディングインデックス

    Returns:
        nn.Module: モデル
    """
    model_type = model_type.lower()

    if model_type == 'rnn':
        return SentimentRNN(vocab_size, embedding_dim, hidden_dim, output_dim,
                           n_layers, bidirectional, dropout, pad_idx)
    elif model_type == 'lstm':
        return SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim,
                            n_layers, bidirectional, dropout, pad_idx)
    elif model_type == 'gru':
        return SentimentGRU(vocab_size, embedding_dim, hidden_dim, output_dim,
                           n_layers, bidirectional, dropout, pad_idx)
    elif model_type == 'ugrnn':
        return SentimentUGRNN(vocab_size, embedding_dim, hidden_dim, output_dim,
                             n_layers, bidirectional, dropout, pad_idx)
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from: rnn, lstm, gru, ugrnn")


def count_parameters(model):
    """
    モデルの学習可能パラメータ数をカウント
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


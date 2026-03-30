"""Mô hình LSTM + Multi-Head Self-Attention dự đoán quỹ đạo tàu biển."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ══════════════════════════════════════════════════════════════════════════════
# Positional Encoding
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Mã hóa vị trí dạng sin — thêm thông tin vị trí thời gian."""

    def __init__(self, d_model: int, max_len: int = 32, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════════════
# ShipLSTM — attention cộng đầu (single-head) giữ nguyên để tham khảo
# ══════════════════════════════════════════════════════════════════════════════

class ShipLSTM(nn.Module):
    """LSTM hai lớp với cơ chế attention cộng đầu (single-head).

    Kiến trúc::

        Đầu vào  (B, T, F)
          ↓
        LSTM   (B, T, H)   — 2 lớp, dropout giữa các lớp
          ↓
        Attn   (B, T, 1)   — điểm số mỗi bước thời gian → softmax → vector ngữ cảnh
          ↓
        FC     (B, 2)      — chiếu tới (X_norm, Y_norm)

    Các tham số
    ----------
    input_size : int
        Số đặc trưng mỗi bước thời gian (mặc định 9).
    hidden_size : int
        Kích thước ẩn của LSTM (mặc định 256).
    num_layers : int
        Số lớp LSTM xếp chồng (mặc định 2).
    dropout : float
        Dropout giữa các lớp LSTM (mặc định 0.2).
    output_size : int
        Kích thước đầu ra dự đoán (mặc định 2).
    """

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền xuôi.

        Các tham số
        ----------
        x : Tensor, kích thước (B, T, F)

        Trả về
        -------
        Tensor, kích thước (B, output_size)
        """
        out, _ = self.lstm(x)                        # (B, T, H)
        weights = F.softmax(self.attn(out), dim=1)   # (B, T, 1)
        context = (weights * out).sum(dim=1)          # (B, H)
        return self.fc(context)                       # (B, 2)


# ══════════════════════════════════════════════════════════════════════════════
# ShipLSTMAttention — LSTM + Multi-Head Self-Attention (phiên bản nâng cấp)
# ══════════════════════════════════════════════════════════════════════════════

class ShipLSTMAttention(nn.Module):
    """LSTM hai lớp theo sau bởi Multi-Head Self-Attention với các kết nối
    residual và LayerNorm (dạng pre-norm theo phong cách transformer).

    Kiến trúc::

        Đầu vào  (B, T, F)
          ↓
        Chiếu đầu vào  Linear(F → H)
          ↓
        Mã hóa vị trí (sinusoidal)
          ↓
        LSTM × 2 lớp, dropout  →  (B, T, H)
          ↓
        ResidualAdd(LayerNorm → MultiHeadSelfAttn → Dropout)
          ↓
        ResidualAdd(LayerNorm → FFN(GELU) → Dropout)
          ↓
        Attention pooling (điểm số → softmax → tổng có trọng số)
          ↓
        FC  H → output_size(2)

    Multi-head attention cho phép mô hình cân nhắc mỗi bước thời gian với tất
    cả các bước khác — điều này rất quan trọng cho quỹ đạo tàu khi cả các
    manh manoeuvre gần đây (quay lái) và các waypoint xa đều mang tín hiệu dự đoán.
    """

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 2,
        num_heads: int = 4,
        ffwd_dim: int = 512,
        attention_dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Chiếu đặc trưng thô → kích thước mô hình
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Mã hóa vị trí dạng sinusoidal
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=32, dropout=dropout)

        # Khung LSTM
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Transformer layer dạng pre-norm: LayerNorm → SelfAttn → Add → LayerNorm → FFN → Add
        attn_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffwd_dim,
            dropout=attention_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm: ổn định hơn cho deep stacks
        )
        self.attn_block = nn.TransformerEncoder(attn_layer, num_layers=1)

        # Attention pooling: điểm số học được mỗi bước thời gian → softmax → tổng có trọng số
        self.pool_query = nn.Linear(hidden_size, 1)
        self.pool_dropout = nn.Dropout(dropout)

        # Chiếu đầu ra
        self.fc = nn.Linear(hidden_size, output_size)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform cho các lớp linear, orthogonal cho các trọng số recurrent LSTM."""
        for m in [self.input_proj, self.fc, self.pool_query]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)  # forget gate bias = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Các tham số
        ----------
        x : torch.Tensor, kích thước (B, T, input_size)

        Trả về
        -------
        torch.Tensor, kích thước (B, output_size)
        """
        # Chiếu & mã hóa vị trí
        h = self.pos_encoder(self.input_proj(x))   # (B, T, H)

        # Đặc trưng LSTM
        lstm_out, _ = self.lstm(h)                  # (B, T, H)

        # Multi-Head Self-Attention (kèm residual)
        attn_out = self.attn_block(lstm_out)       # (B, T, H)

        # Attention pooling theo thời gian
        scores = self.pool_query(attn_out)          # (B, T, 1)
        weights = self.pool_dropout(F.softmax(scores, dim=1))
        context = (weights * attn_out).sum(dim=1)   # (B, H)

        return self.fc(context)                     # (B, output_size)


# ══════════════════════════════════════════════════════════════════════════════
# Model factory
# ══════════════════════════════════════════════════════════════════════════════

def build_model(cfg: dict) -> nn.Module:
    """Khởi tạo mô hình từ phần ``model`` của config.

    Các giá trị model.type
    ------------------
    lstm_attention       → ShipLSTM (attention cộng đầu single-head)
    lstm_attention_tube  → ShipLSTMAttention (Multi-Head Self-Attention)
    """
    m = cfg.get("model", {})
    model_type = m.get("type", "lstm_attention")

    if model_type == "lstm_attention_tube":
        return ShipLSTMAttention(
            input_size=m.get("input_size", 9),
            hidden_size=m.get("hidden_size", 256),
            num_layers=m.get("num_layers", 2),
            dropout=m.get("dropout", 0.2),
            output_size=m.get("output_size", 2),
            num_heads=m.get("num_heads", 4),
            ffwd_dim=m.get("ffwd_dim", 512),
            attention_dropout=m.get("attention_dropout", 0.1),
        )

    # Mặc định: ShipLSTM gốc (attention cộng đầu single-head)
    return ShipLSTM(
        input_size=m.get("input_size", 9),
        hidden_size=m.get("hidden_size", 256),
        num_layers=m.get("num_layers", 2),
        dropout=m.get("dropout", 0.2),
        output_size=m.get("output_size", 2),
    )

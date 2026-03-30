"""Mô hình LSTM + Attention + Embedding loại tàu.

Kiến trúc:
    Đầu vào  (B, T, 20)        ← 9 đặc trưng AIS + 11 đặc trưng môi trường
      ↓
    Chiếu đầu vào Linear(20 → H)
      ↓
    Mã hóa vị trí (Positional Encoding)
      ↓
    LSTM × num_layers         → (B, T, H)
      ↓
    Tự chú ý đa đầu (Multi-Head Self-Attention)  → (B, T, H)
      ↓
    Gom chú ý (Attention pooling)  → (B, H)      (vector ngữ cảnh)
      ↓ nối (concat)
    Embedding loại tàu(n_types → emb_dim)  → (B, emb_dim)
      ↓
    FC: (H + emb_dim) → hidden_fc → 2
      ↓
    Đầu ra (B, 2)             ← (X_norm, Y_norm)

Khi suy luận: vessel_type khác nhau → embedding khác nhau
→ mô hình tạo ra quỹ đạo khác nhau cho mỗi loại tàu.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_VESSEL_TYPES = 6   # 0=Mặc định,1=Tàu dầu,2=Tàu hàng,3=Tàu khách,4=Tàu cá,5=Tàu kéo


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 32, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class ShipLSTMEnvAttention(nn.Module):
    """LSTM + Chú ý đa đầu + Embedding loại tàu.

    Tham số
    ----------
    input_size   : int   Số đặc trưng mỗi bước thời gian (mặc định 20 = 9 AIS + 11 môi trường)
    hidden_size  : int   Kích thước ẩn LSTM (mặc định 256)
    num_layers   : int   Số tầng LSTM (mặc định 2)
    dropout      : float Dropout giữa các tầng LSTM (mặc định 0.2)
    num_heads    : int   Số đầu MHA (mặc định 4)
    ffwd_dim     : int   Kích thước ẩn FFN trong TransformerEncoderLayer (mặc định 512)
    attn_dropout : float Dropout bên trong attention (mặc định 0.1)
    n_vessel_types: int  Số lớp loại tàu (mặc định 6)
    emb_dim      : int   Kích thước embedding loại tàu (mặc định 16)
    """

    def __init__(
        self,
        input_size:    int   = 20,
        hidden_size:   int   = 256,
        num_layers:    int   = 2,
        dropout:       float = 0.2,
        num_heads:     int   = 4,
        ffwd_dim:      int   = 512,
        attn_dropout:  float = 0.1,
        n_vessel_types: int  = N_VESSEL_TYPES,
        emb_dim:       int   = 16,
        output_size:   int   = 2,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        self.hidden_size    = hidden_size
        self.emb_dim        = emb_dim
        self.n_vessel_types = n_vessel_types

        # ── Chiếu đầu vào ──────────────────────────────────────────────────
        self.input_proj = nn.Linear(input_size, hidden_size)

        # ── Mã hóa vị trí ───────────────────────────────────────────────────
        self.pos_enc = PositionalEncoding(hidden_size, max_len=32, dropout=dropout)

        # ── Xương sống LSTM ──────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── Tự chú ý đa đầu (chuẩn hóa trước) ─────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=ffwd_dim,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.attn_block = nn.TransformerEncoder(enc_layer, num_layers=1)

        # ── Gom chú ý (Attention pooling) ───────────────────────────────────
        self.pool_query   = nn.Linear(hidden_size, 1)
        self.pool_dropout = nn.Dropout(dropout)

        # ── Embedding loại tàu ────────────────────────────────────────────────
        self.vessel_emb = nn.Embedding(n_vessel_types, emb_dim, padding_idx=None)

        # ── Đầu ra: ngữ cảnh + embedding → vị trí ───────────────────────────
        fc_in = hidden_size + emb_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_in, fc_in // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_in // 2, output_size),
        )

        self._init_weights()

    # ── Khởi tạo trọng số ───────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p.data)
            elif "bias" in name:
                nn.init.zeros_(p.data)
                n = p.size(0)
                p.data[n // 4 : n // 2].fill_(1.0)   # bias cổng quên = 1
        nn.init.normal_(self.vessel_emb.weight, mean=0.0, std=0.1)

    # ── Lan truyền thuận ────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        vessel_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tham số
        ----------
        x           : (B, T, input_size)   Đặc trưng AIS + môi trường
        vessel_type : (B,)  int64  mã loại tàu (0-5)

        Trả về
        -------
        (B, 2)  Dự đoán vị trí tiếp theo (X_norm, Y_norm)
        """
        # Chiếu + mã hóa vị trí
        h = self.pos_enc(self.input_proj(x))     # (B, T, H)

        # LSTM
        lstm_out, _ = self.lstm(h)               # (B, T, H)

        # Tự chú ý đa đầu
        attn_out = self.attn_block(lstm_out)     # (B, T, H)

        # Gom chú ý → vector ngữ cảnh
        scores  = self.pool_query(attn_out)      # (B, T, 1)
        weights = self.pool_dropout(F.softmax(scores, dim=1))
        context = (weights * attn_out).sum(dim=1) # (B, H)

        # Embedding loại tàu
        emb = self.vessel_emb(vessel_type)       # (B, emb_dim)

        # Nối + FC
        out = self.fc(torch.cat([context, emb], dim=1))  # (B, 2)
        return out


class ShipLSTMEnvAttentionMultiStep(nn.Module):
    """LSTM + Chú ý đa đầu + Embedding loại tàu — Dự đoán 5 bước.

    Cùng bộ mã hóa với ShipLSTMEnvAttention, đầu ra được thay đổi để
    dự đoán đồng thời n_steps vị trí tương lai.

    Kích thước đầu ra: (B, n_steps, 2)
    Khi suy luận: embedding vessel_type khác nhau theo loại → quỹ đạo khác nhau.
    """

    def __init__(
        self,
        input_size:     int   = 20,
        hidden_size:    int   = 256,
        num_layers:     int   = 2,
        dropout:        float = 0.2,
        num_heads:      int   = 4,
        ffwd_dim:       int   = 512,
        attn_dropout:   float = 0.1,
        n_vessel_types: int   = N_VESSEL_TYPES,
        emb_dim:        int   = 16,
        n_steps:        int   = 5,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.n_steps     = n_steps
        self.hidden_size = hidden_size
        self.emb_dim     = emb_dim

        # ── Bộ mã hóa (giống hệt mô hình 1 bước) ──────────────────────────
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_enc    = PositionalEncoding(hidden_size, max_len=32, dropout=dropout)
        self.lstm       = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=ffwd_dim, dropout=attn_dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.attn_block   = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.pool_query   = nn.Linear(hidden_size, 1)
        self.pool_dropout = nn.Dropout(dropout)

        # ── Embedding loại tàu ────────────────────────────────────────────────
        self.vessel_emb = nn.Embedding(n_vessel_types, emb_dim)

        # ── Đầu ra nhiều bước ─────────────────────────────────────────────────
        # ngữ cảnh(H) + emb(E) → FC → n_steps*2 → reshape (B, n_steps, 2)
        fc_in = hidden_size + emb_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_in, fc_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_in, n_steps * 2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p.data)
            elif "bias" in name:
                nn.init.zeros_(p.data)
                n = p.size(0)
                p.data[n // 4 : n // 2].fill_(1.0)
        nn.init.normal_(self.vessel_emb.weight, 0.0, 0.1)

    def forward(
        self,
        x: torch.Tensor,
        vessel_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tham số
        ----------
        x           : (B, T, 20)   Đặc trưng AIS + môi trường
        vessel_type : (B,)  int64

        Trả về
        -------
        (B, n_steps, 2)   Dự đoán vị trí cho mỗi bước tương lai
        """
        h = self.pos_enc(self.input_proj(x))      # (B, T, H)
        lstm_out, _ = self.lstm(h)                 # (B, T, H)
        attn_out    = self.attn_block(lstm_out)    # (B, T, H)
        scores      = self.pool_query(attn_out)    # (B, T, 1)
        weights     = self.pool_dropout(F.softmax(scores, dim=1))
        context     = (weights * attn_out).sum(1)  # (B, H)

        emb = self.vessel_emb(vessel_type)         # (B, E)
        flat = self.fc(torch.cat([context, emb], dim=1))   # (B, n_steps*2)
        return flat.view(-1, self.n_steps, 2)              # (B, n_steps, 2)

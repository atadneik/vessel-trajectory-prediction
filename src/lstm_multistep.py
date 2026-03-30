"""ShipLSTMMultiStep — LSTM+Attention model for multi-step trajectory prediction.

File mới, không sửa lstm.py gốc.

Kiến trúc giống ShipLSTM (800K params, 1-step) nhưng output layer
predict n_steps bước cùng lúc:

    Input  (B, T, F)
      ↓
    LSTM × 2 layers + additive attention
      ↓
    FC  H → n_steps × 2  →  reshape (B, n_steps, 2)

Tube loss áp dụng trên từng bước với weight giảm dần theo thời gian
(bước gần hơn → weight cao hơn).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShipLSTMMultiStep(nn.Module):
    """Two-layer LSTM + additive attention, predict n_steps future positions.

    Parameters
    ----------
    input_size  : features per timestep (default 9)
    hidden_size : LSTM hidden dim (default 256)
    num_layers  : stacked LSTM layers (default 2)
    dropout     : inter-layer dropout (default 0.2)
    n_steps     : number of future steps to predict (default 5)
    """

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_steps: int = 5,
    ):
        super().__init__()
        self.n_steps = n_steps

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden_size, 1)
        # output: n_steps * 2 (X_norm, Y_norm for each step)
        self.fc = nn.Linear(hidden_size, n_steps * 2)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)   # forget gate bias = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, input_size)

        Returns
        -------
        (B, n_steps, 2)  — predicted [X_norm, Y_norm] for each future step
        """
        out, _ = self.lstm(x)                        # (B, T, H)
        weights = F.softmax(self.attn(out), dim=1)   # (B, T, 1)
        context = (weights * out).sum(dim=1)          # (B, H)
        flat = self.fc(context)                       # (B, n_steps*2)
        return flat.view(-1, self.n_steps, 2)         # (B, n_steps, 2)


def build_multistep_model(cfg: dict) -> ShipLSTMMultiStep:
    """Build ShipLSTMMultiStep from config dict."""
    m = cfg.get("model", {})
    return ShipLSTMMultiStep(
        input_size=m.get("input_size", 9),
        hidden_size=m.get("hidden_size", 256),
        num_layers=m.get("num_layers", 2),
        dropout=m.get("dropout", 0.2),
        n_steps=m.get("n_steps", 5),
    )

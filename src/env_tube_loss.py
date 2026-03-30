import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

# --- CHỈ SỐ CỘT TRONG TENSOR (23 CỘT) ---
_SOG = 0
_H_SIN = 1
_H_COS = 2
_X = 5
_Y = 6

_WAVE_H = 9
_WAVE_DIR_SIN = 10
_WAVE_DIR_COS = 11
_WAVE_T = 12

_WIND_SPD = 13
_WIND_DIR_SIN = 14
_WIND_DIR_COS = 15
_WIND_GUST = 16

_CURR_SPD = 17
_CURR_DIR_SIN = 18
_CURR_DIR_COS = 19

# --- CỘT HÌNH HỌC TÀU ---
_L = 20
_B = 21
_T = 22


class PhysicalEnvLoss(nn.Module):
    """
    Hàm mất mát ống hành lang nhận biết môi trường vật lý (Loss V2).
    Tính ranh giới hành lang ngang và dọc theo đơn vị MÉT dựa trên
    vật lý tàu, sau đó chuyển sang ĐƠN VỊ CHUẨN HÓA để so sánh với dự đoán.

    Hệ tọa độ:
      - X/Y trong parquet = độ. Chuyển đổi độ dịch từ mét → đơn vị chuẩn hóa
        theo: 1 đơn_vị_chuẩn_hóa ≈ 1000 m (xấp xỉ, phụ thuộc dữ liệu).
        Được hiệu chuẩn sao cho std độ dịch GT ≈ 0.1–0.2 đơn vị chuẩn hóa
        và độ lệch ngang điển hình < 0.2 đơn vị chuẩn hóa.
      - SOG = hải lý/giờ thô (không chuẩn hóa trong parquet).
      - L, B, T = mét thô từ parquet thông tin tàu.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        # Trọng số cho tổng Loss
        self.w_lat = cfg.get("w_lat", 1.0)
        self.w_fwd = cfg.get("w_fwd", 1.0)
        self.w_prog = cfg.get("w_prog", 0.5)
        self.kappa = cfg.get("kappa", 5.0)
        self.dt = cfg.get("delta_t", 66.0)

        # Hệ số chuyển đổi đơn vị chuẩn hóa → mét để so sánh ranh giới
        # 1 đơn_vị_chuẩn_hóa ≈ 1000m dựa trên thống kê độ dịch GT
        self.norm_to_m = cfg.get("norm_to_m", 1000.0)

        # ─── THAM SỐ HÀNH LANG (đơn vị MÉT) ────────────────────────────────
        # y_base: nửa chiều rộng ngang cơ sở (mét)
        #   Dựa trên dữ liệu: trung vị |độ lệch ngang| ≈ 100m → y_base ≈ 0.1*norm_to_m = 100m
        self.y_base = cfg.get("y_base", 100.0)

        # ─── HỆ SỐ QUY MÔ VẬT LÝ TÀU ──────────────────────────────────────
        # Phục hồi giá trị gốc của đặc trưng môi trường
        self.scale_wave_h = cfg.get("scale_wave_h", 15.0)
        self.scale_wave_t = cfg.get("scale_wave_t", 25.0)
        self.scale_wind   = cfg.get("scale_wind", 40.0)
        self.scale_gust   = cfg.get("scale_gust", 50.0)
        self.scale_curr   = cfg.get("scale_curr", 5.0)

        # Hình học tàu: đã ở đơn vị mét thô từ parquet — không cần chia tỉ lệ
        self.scale_L = cfg.get("scale_L", 1.0)
        self.scale_B = cfg.get("scale_B", 1.0)
        self.scale_T = cfg.get("scale_T", 1.0)

        # ─── HỆ SỐ VẬT LÝ HỌC ĐƯỢC ─────────────────────────────────────────
        self.alpha_lat_wind = nn.Parameter(torch.tensor(0.1))
        self.alpha_lon_wind = nn.Parameter(torch.tensor(0.1))

        self.alpha_lat_curr = nn.Parameter(torch.tensor(1.0))
        self.alpha_lon_curr = nn.Parameter(torch.tensor(1.0))

        self.alpha_lat_wave = nn.Parameter(torch.tensor(0.5))
        self.alpha_yaw_wave = nn.Parameter(torch.tensor(0.5))
        self.alpha_wave_res = nn.Parameter(torch.tensor(0.1))

        # k_brake: hằng số gia tốc phanh (m/s² trên mỗi đơn vị k_brake / Delta)
        self.k_brake = nn.Parameter(torch.tensor(1000.0))

    def _softplus_penalty(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.kappa * x) / self.kappa

    def forward(self, pred: torch.Tensor, x_batch: torch.Tensor,
                s_max_norm: float, vessel_type: str = "Default") -> Tuple[torch.Tensor, Dict[str, float]]:

        device = pred.device

        # Ép các Alpha luôn dương
        a_lat_wind = F.softplus(self.alpha_lat_wind)
        a_lon_wind = F.softplus(self.alpha_lon_wind)
        a_lat_curr = F.softplus(self.alpha_lat_curr)
        a_lon_curr = F.softplus(self.alpha_lon_curr)
        a_lat_wave = F.softplus(self.alpha_lat_wave)
        a_yaw_wave = F.softplus(self.alpha_yaw_wave)
        a_wave_res = F.softplus(self.alpha_wave_res)
        k_brk      = F.softplus(self.k_brake)

        # =====================================================================
        # KHỐI 1: KHÔNG TÍNH ĐẠO HÀM
        # Chỉ tách và phục hồi giá trị gốc dữ liệu thô
        # =====================================================================
        with torch.no_grad():
            f9 = x_batch[:, -1, :]  # bước thời gian cuối cùng

            # ── ĐỘNG HỌC (SOG: hải lý/giờ thô → m/s)
            v0 = f9[:, _SOG] * 0.5144  # hải lý/giờ → m/s
            h_sin, h_cos = f9[:, _H_SIN], f9[:, _H_COS]

            # ── HÌNH HỌC TÀU (mét thô → mét thực tế)
            L_raw = f9[:, _L]   # giá trị thô từ parquet
            B_raw = f9[:, _B]
            T_raw = f9[:, _T]
            L = L_raw * self.scale_L + 1e-6
            B = B_raw * self.scale_B + 1e-6
            T = T_raw * self.scale_T + 1e-6
            Cb = 0.7  # hệ số béo mặc định

            # ── MÔI TRƯỜNG (phục hồi giá trị gốc)
            h_wave = f9[:, _WAVE_H] * self.scale_wave_h
            t_wave = f9[:, _WAVE_T] * self.scale_wave_t + 1e-6
            w_sin, w_cos = f9[:, _WAVE_DIR_SIN], f9[:, _WAVE_DIR_COS]

            v_wind = f9[:, _WIND_SPD] * self.scale_wind
            v_gust = f9[:, _WIND_GUST] * self.scale_gust
            wind_sin, wind_cos = f9[:, _WIND_DIR_SIN], f9[:, _WIND_DIR_COS]

            v_curr = f9[:, _CURR_SPD] * self.scale_curr
            c_sin, c_cos = f9[:, _CURR_DIR_SIN], f9[:, _CURR_DIR_COS]

            # ── GÓC TẠT TƯƠNG ĐỐI
            delta_wind_sin = -wind_sin * h_cos + wind_cos * h_sin
            delta_wind_cos = -wind_cos * h_cos - wind_sin * h_sin

            delta_curr_sin = c_sin * h_cos - c_cos * h_sin
            delta_curr_cos = c_cos * h_cos + c_sin * h_sin

            delta_wave_sin = w_sin * h_cos - w_cos * h_sin

        # =====================================================================
        # KHỐI 2: TÍNH TOÁN CÓ ĐẠO HÀM — các alpha học được sẽ được cập nhật
        # =====================================================================

        # ── Hệ số C (hệ số cản lai ghép) ────────────────────────────────────
        Delta = L * B * T * Cb * 1.025
        sqrt_Delta = torch.sqrt(Delta)

        C_lat_wind = a_lat_wind * (L / sqrt_Delta)
        C_lon_wind = a_lon_wind * (B / sqrt_Delta)

        C_lat_curr = a_lat_curr * (1.0 / (B * Cb))
        C_lon_curr = a_lon_curr * (1.0 / (L * Cb))

        C_lat_wave = a_lat_wave * (B / L) * (1.0 / sqrt_Delta)
        C_yaw_wave = a_yaw_wave * (B / L) * (1.0 / sqrt_Delta)
        C_wave_res = a_wave_res * ((B**2) / L) * (1.0 / sqrt_Delta)

        # Gia tốc phanh: k_brake / (Delta * T)
        a_max_t = k_brk / (Delta * T)

        # ── Vận tốc môi trường tổng hợp ─────────────────────────────────────
        V_lat_env = (C_lat_wind * (v_wind + v_gust) * delta_wind_sin +
                     C_lat_curr * v_curr * delta_curr_sin +
                     C_lat_wave * h_wave * delta_wave_sin)

        V_lon_env = (C_lon_wind * (v_wind + v_gust) * delta_wind_cos +
                     C_lon_curr * v_curr * delta_curr_cos)

        # ── Ranh giới dọc (tiến) ─────────────────────────────────────────────
        # s_max: vận tốc tối đa (v_max, m/s) * dt + lực môi trường
        # s_max_norm truyền vào = v_max tính bằng m/s (từ vm[:, 0])
        s_max_new = s_max_norm * self.dt + V_lon_env * self.dt - (C_wave_res * h_wave**2) * self.dt
        s_max_new = torch.clamp(s_max_new, min=1.0)  # sàn: ít nhất 1m

        # s_min: quãng đường khi phanh khẩn cấp từ SOG hiện tại (v0)
        # Không dùng v_max vì v_max là tốc độ tối đa, không phải tốc độ hiện tại
        s_min_new = v0 * self.dt - 0.5 * a_max_t * self.dt**2 + V_lon_env * self.dt
        # sàn: ít nhất 5% của s_max_new
        s_min_new = torch.clamp(s_min_new, min=0.05 * s_max_new)

        # ── Ranh giới ngang ───────────────────────────────────────────────────
        dy_env = V_lat_env * self.dt

        # y_base có thể được điều chỉnh bởi sóng (hiệu ứng lệch hướng mũi)
        y_base_new = self.y_base + C_yaw_wave * h_wave * (1.0 + h_wave / (1.56 * t_wave**2))

        # Bất đối xứng: môi trường đẩy tàu sang một bên
        y_L_tgt = y_base_new + torch.clamp(dy_env, min=0.0)   # ranh giới trái (mét)
        y_R_tgt = y_base_new + torch.clamp(-dy_env, min=0.0)  # ranh giới phải (mét)

        # =====================================================================
        # KHỐI 3: CHUYỂN ĐỔI ĐƠN VỊ VÀ TÍNH LOSS
        # =====================================================================

        # pred = (X_norm, Y_norm) tọa độ tuyệt đối
        # p0 = (X_t9, Y_t9) = vị trí cuối cùng đã biết theo đơn vị chuẩn hóa
        p0 = f9[:, _X:_Y + 1]
        disp = pred - p0  # (B, 2) độ dịch theo đơn vị chuẩn hóa

        # Chiếu thành thành phần dọc trục (s) và vuông góc trục (d)
        e_par  = torch.stack([h_cos, h_sin],  dim=1)    # véc-tơ đơn vị hướng tiến
        e_perp = torch.stack([-h_sin, h_cos], dim=1)   # véc-tơ đơn vị hướng ngang

        s_pred = (disp * e_par).sum(dim=1)   # độ dịch dọc (đơn vị chuẩn hóa)
        d_pred = (disp * e_perp).sum(dim=1)  # độ dịch ngang (đơn vị chuẩn hóa)

        # Chuyển ranh giới hành lang từ MÉT → ĐƠN VỊ CHUẨN HÓA
        # 1 đơn vị chuẩn hóa ≈ norm_to_m mét
        y_L_norm = y_L_tgt / self.norm_to_m
        y_R_norm = y_R_tgt / self.norm_to_m
        s_max_norm_units = s_max_new / self.norm_to_m
        s_min_norm_units = s_min_new / self.norm_to_m

        # ── Phạt ngang ────────────────────────────────────────────────────────
        pen_lat_L = self._softplus_penalty(d_pred - y_L_norm)
        pen_lat_R = self._softplus_penalty(y_R_norm - d_pred)
        loss_lat = pen_lat_L**2 + pen_lat_R**2

        # ── Phạt dọc ─────────────────────────────────────────────────────────
        pen_fwd = self._softplus_penalty(s_pred - s_max_norm_units)
        loss_fwd = pen_fwd**2

        pen_prog = self._softplus_penalty(s_min_norm_units - s_pred)
        loss_prog = pen_prog**2

        total_loss = (self.w_lat * loss_lat.mean() +
                      self.w_fwd * loss_fwd.mean() +
                      self.w_prog * loss_prog.mean())

        info = {
            "loss_lat": loss_lat.mean().item(),
            "loss_fwd": loss_fwd.mean().item(),
            "loss_prog": loss_prog.mean().item(),
            "a_lat_wind": a_lat_wind.item(),
            "a_lat_curr": a_lat_curr.item(),
            "a_wave_res": a_wave_res.item(),
        }
        return total_loss, info

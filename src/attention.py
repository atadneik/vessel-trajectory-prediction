"""Các module ràng buộc hình học: ống dẫn (Tube/corridor) và mất mát Ellipse.

Các module này triển khai các penalty hình học dựa trên vật lý nhằm khuyến khích
các dự đoán nằm trong các hành lang chuyển động hợp lý. Chúng hoạt động trong
không gian tọa độ đã được chuẩn hóa mà LSTM sử dụng.

Cả hai module đều chia sẻ cùng một quy ước chỉ mục phụ trợ (``IDX``) cho tensor
metadata bổ sung theo từng mẫu.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

# ── Hằng số dùng chung ──────────────────────────────────────────────────────

IDX = {
    "dt": 0, "sog": 1, "hsin": 2, "hcos": 3,
    "x0": 4, "y0": 5, "turn": 6, "a_t": 7,
    "a_t_max": 8, "a_n_max": 9, "v_max": 10, "hdiff": 11,
}
KTS2MPS = 0.514444


# ── Hàm hỗ trợ dùng chung cho cả hai ràng buộc ──────────────────────────

@torch.no_grad()
def axes_motion(aux: torch.Tensor):
    """Tính các vector cơ sở trong khung chuyển động (e_par, e_perp)."""
    c0, s0 = aux[:, IDX["hcos"]], aux[:, IDX["hsin"]]
    d = aux[:, IDX["hdiff"]]
    cd, sd = torch.cos(d), torch.sin(d)
    c = c0 * cd - s0 * sd
    s = s0 * cd + c0 * sd
    e_par = torch.stack([c, s], 1)
    e_perp = torch.stack([-s, c], 1)
    return e_par, e_perp


@torch.no_grad()
def project_s(
    xy: torch.Tensor,
    p0: torch.Tensor,
    e_par0: torch.Tensor,
    e_perp0: torch.Tensor,
    s_max: float,
    a_curve: float,
    iters: int = 4,
    eps: float = 1e-8,
):
    """Chiếu Newton lên đường tâm cong."""
    u = torch.sum((xy - p0) * e_par0, dim=1).clamp(0, s_max)
    for _ in range(iters):
        t = u / (s_max + eps)
        dlt = a_curve * t * (1 - t)
        d1 = a_curve * (1 - 2 * t) / (s_max + eps)
        d2 = -2.0 * a_curve / (s_max**2 + eps)
        pc = p0 + u.unsqueeze(1) * e_par0 + dlt.unsqueeze(1) * e_perp0
        pc1 = e_par0 + d1.unsqueeze(1) * e_perp0
        pc2 = d2 * e_perp0
        r = xy - pc
        f1 = -2.0 * torch.sum(r * pc1, dim=1)
        f2 = 2.0 * torch.sum(pc1 * pc1, dim=1) - 2.0 * torch.sum(r * pc2, dim=1)
        u = (u - f1 / (f2 + eps)).clamp(0, s_max)

    t = u / (s_max + eps)
    dlt = a_curve * t * (1 - t)
    d1 = a_curve * (1 - 2 * t) / (s_max + eps)
    pc1 = e_par0 + d1.unsqueeze(1) * e_perp0
    epar = pc1 / (pc1.norm(dim=1, keepdim=True) + 1e-8)
    eprp = torch.stack([-epar[:, 1], epar[:, 0]], dim=1)
    p_c = p0 + u.unsqueeze(1) * e_par0 + dlt.unsqueeze(1) * e_perp0
    y_pf = torch.sum((xy - p_c) * eprp, dim=1)
    return u, y_pf, epar, eprp


@torch.no_grad()
def tau_and_dir(aux: torch.Tensor, params):
    """Tính cường độ quay đã chuẩn hóa (tau) và dấu hướng quay."""
    turn = aux[:, IDX["turn"]]
    a_tn = torch.clamp(-aux[:, IDX["a_t"]], min=0)
    tau = torch.maximum(
        torch.sigmoid((turn.abs() - params.thr_w) / params.s_w),
        torch.sigmoid((a_tn - params.thr_at) / params.s_at),
    )
    dir_s = torch.sign(torch.where(turn == 0, torch.ones_like(turn), turn))
    return tau, dir_s


# ══════════════════════════════════════════════════════════════════════════
# RÀNG BUỘC HÀNH LANG ỐNG DẪN (TUBE CORRIDOR)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class TubeCfg:
    """Cấu hình cho ràng buộc hình học kiểu ống dẫn/hành lang."""
    alpha: float = 1.0
    beta: float = 1.0
    bump_gain: Optional[float] = None
    cap_lo: float = 0.30
    cap_hi: float = 1.60
    alpha_curve: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "TubeCfg":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@torch.no_grad()
def rho_ab(t: torch.Tensor, alpha: float = 1.0, beta: float = 1.0, eps: float = 1e-8):
    """Hàm trọng số dạng phân bố Beta đã chuẩn hóa về [0, 1]."""
    t = t.clamp(0, 1)
    t_peak = alpha / (alpha + beta + eps)
    peak = (t_peak**alpha) * ((1 - t_peak)**beta)
    val = (t**alpha) * ((1 - t)**beta)
    return val / (peak + eps)


def rho_ab_integral(alpha: float, beta: float, eps: float = 1e-12) -> float:
    t_peak = alpha / (alpha + beta + eps)
    peak = (t_peak**alpha) * ((1 - t_peak)**beta) + eps
    num = math.gamma(alpha + 1) * math.gamma(beta + 1) / math.gamma(alpha + beta + 2)
    return num / peak


@torch.no_grad()
def tube_inside_and_area(
    xy: torch.Tensor, aux: torch.Tensor, params, gamma_m: torch.Tensor,
    scaler_xy, cfg: TubeCfg,
):
    """Tính mask inside theo từng mẫu và diện tích ống dẫn."""
    e_par0, e_perp0 = axes_motion(aux)
    p0 = torch.stack([aux[:, IDX["x0"]], aux[:, IDX["y0"]]], 1)
    s_cp, y_pf, e_par, _ = project_s(xy, p0, e_par0, e_perp0, params.s_max, cfg.alpha_curve)

    dt, sog = aux[:, IDX["dt"]], aux[:, IDX["sog"]]
    sx, sy = float(scaler_xy.scale_[0]), float(scaler_xy.scale_[1])
    c, s = e_par[:, 0], e_par[:, 1]
    scale_par = torch.sqrt((sx * c)**2 + (sy * s)**2) + 1e-8

    tau, dir_s = tau_and_dir(aux, params)
    s_tgt_norm = (gamma_m * sog * KTS2MPS * dt) / scale_par
    s_cap = torch.clamp(
        (1.0 + 0.30 * tau) * s_tgt_norm,
        cfg.cap_lo * params.s_max, cfg.cap_hi * params.s_max,
    )

    t_cap = torch.clamp(s_cp / (s_cap + 1e-8), 0, 1)
    rho = rho_ab(t_cap, cfg.alpha, cfg.beta)
    B = params.B_max * rho * (cfg.bump_gain if cfg.bump_gain else 1.0)

    eps_b = 0.1 * params.r_base
    yL = params.r_base + tau * torch.where(dir_s > 0, B, torch.full_like(B, eps_b))
    yR = params.r_base + tau * torch.where(dir_s < 0, B, torch.full_like(B, eps_b))

    inside = (
        (y_pf <= yL + 1e-6) & (-y_pf <= yR + 1e-6)
        & (s_cp >= -1e-6) & (s_cp <= s_cap + 1e-6)
    )

    Iab = rho_ab_integral(cfg.alpha, cfg.beta)
    area = s_cap * (
        2.0 * params.r_base
        + tau * (0.1 * params.r_base)
        + tau * params.B_max * (cfg.bump_gain if cfg.bump_gain else 1.0) * Iab
    )

    cache = {"s_cp": s_cp, "y_pf": y_pf, "s_cap": s_cap, "yL": yL, "yR": yR}
    return inside, area, cache


# ══════════════════════════════════════════════════════════════════════════
# RÀNG BUỘC ELLIPSE
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class EllipseCfg:
    """Cấu hình cho ràng buộc hình học ellipse/superellipse."""
    kappa_s: float = 1.08
    kappa_y: float = 1.12
    beta_tau: float = 0.35
    p: float = 2.0
    p_turn: float = 3.6
    b_min_frac: float = 0.10
    use_s_cap: bool = True
    asym_gain: float = 1.10
    tau_gain: float = 0.30
    floor_frac: float = 0.60
    ceil_frac: float = 1.50
    alpha_curve: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "EllipseCfg":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@torch.no_grad()
def ellipse_inside_and_area(
    xy: torch.Tensor, aux: torch.Tensor, params, gamma_m: torch.Tensor,
    scaler_xy, cfg: EllipseCfg,
):
    """Tính mask inside theo từng mẫu và diện tích superellipse."""
    e_par0, e_perp0 = axes_motion(aux)
    p0 = torch.stack([aux[:, IDX["x0"]], aux[:, IDX["y0"]]], 1)
    s_cp, y_pf, e_par, _ = project_s(xy, p0, e_par0, e_perp0, params.s_max, cfg.alpha_curve)

    dt, sog = aux[:, IDX["dt"]], aux[:, IDX["sog"]]
    sx, sy = float(scaler_xy.scale_[0]), float(scaler_xy.scale_[1])
    c, s = e_par[:, 0], e_par[:, 1]
    scale_par = torch.sqrt((sx * c)**2 + (sy * s)**2) + 1e-8

    tau, dir_s = tau_and_dir(aux, params)
    s_tgt_norm = (gamma_m * sog * KTS2MPS * dt) / scale_par

    if cfg.use_s_cap:
        s_cap = torch.clamp(
            (1.0 + cfg.tau_gain * tau) * s_tgt_norm,
            cfg.floor_frac * params.s_max, cfg.ceil_frac * params.s_max,
        )
    else:
        s_cap = torch.full_like(s_tgt_norm, params.s_max)

    a = (cfg.kappa_s * s_cap).clamp_min(1e-6)

    gain = torch.full_like(tau, cfg.asym_gain)
    eps_b = 0.1 * params.r_base
    yL = params.r_base + tau * torch.where(dir_s > 0, gain * params.B_max, torch.full_like(gain, eps_b))
    yR = params.r_base + tau * torch.where(dir_s < 0, gain * params.B_max, torch.full_like(gain, eps_b))
    b_base = 0.5 * (yL + yR)

    ky_eff = cfg.kappa_y * (1.0 + cfg.beta_tau * tau)
    b = torch.maximum(
        ky_eff * b_base,
        torch.full_like(b_base, cfg.b_min_frac * params.r_base),
    ).clamp_min(1e-6)

    delta = 0.5 * (yL - yR)
    p_eff = torch.clamp(cfg.p + (cfg.p_turn - cfg.p) * tau, 2.0, max(cfg.p, cfg.p_turn))

    s_pos = s_cp.clamp_min(0.0)
    r_s = (s_pos / (a + 1e-8)).pow(p_eff)
    r_y = ((y_pf - delta).abs() / (b + 1e-8)).pow(p_eff)
    g = r_s + r_y
    inside = g <= 1.0 + 1e-6

    C = torch.exp(2.0 * torch.lgamma(1.0 + 1.0 / p_eff) - torch.lgamma(1.0 + 2.0 / p_eff))
    area = 4.0 * a * b * C

    cache = {"s_cp": s_cp, "y_pf": y_pf, "a": a, "b": b, "delta": delta, "p_eff": p_eff}
    return inside, area, cache


# ══════════════════════════════════════════════════════════════════════════
# ĐÁNH GIÁ HÌNH HỌC (GEOMETRY)
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_geom(
    xy: torch.Tensor,
    aux: torch.Tensor,
    params,
    gamma_m: torch.Tensor,
    scaler_xy,
    constraint_type: str = "tube",
    tube_cfg: Optional[TubeCfg] = None,
    ellipse_cfg: Optional[EllipseCfg] = None,
    area_ref: Optional[float] = None,
    gamma_sweep: Sequence[float] = (0.90, 0.95, 1.00, 1.05, 1.10),
) -> Dict[str, object]:
    """Đánh giá hình học thống nhất cho ràng buộc tube hoặc ellipse.

    Trả về dict chứa coverage, area, CPA, severity, margin và fairness.
    """
    if constraint_type == "tube":
        cfg = tube_cfg or TubeCfg()
        inside, area, cache = tube_inside_and_area(xy, aux, params, gamma_m, scaler_xy, cfg)
        y_pf = cache["y_pf"]
        r = torch.where(y_pf >= 0, cache["yL"], cache["yR"]).clamp_min(1e-8)
        y_norm = y_pf.abs() / r
    elif constraint_type == "ellipse":
        cfg = ellipse_cfg or EllipseCfg()
        inside, area, cache = ellipse_inside_and_area(xy, aux, params, gamma_m, scaler_xy, cfg)
        s = cache["s_cp"].clamp_min(0.0)
        a = cache["a"].clamp_min(1e-8)
        p = cache["p_eff"].clamp_min(2.0)
        y = (cache["y_pf"] - cache["delta"]).abs()
        xs = (s / a).pow(p)
        g = (1.0 - xs).clamp_min(0.0).pow(1.0 / p)
        b = cache["b"].clamp_min(1e-8)
        y_norm = y / (b * g).clamp_min(1e-8)
    else:
        raise ValueError(f"Unknown constraint_type: {constraint_type}")

    n = inside.numel()
    cov = float(inside.float().mean().item()) if n else float("nan")
    area_mean = float(area.mean().item()) if n else float("nan")
    area_ratio = float(area_mean / area_ref) if (area_ref and area_ref > 0) else float("nan")
    cpa = (cov / area_ratio) if (area_ratio == area_ratio and area_ratio > 0) else float("nan")

    sev = float((y_norm - 1.0).clamp_min(0).mean().item()) if n else float("nan")
    if inside.any():
        m = (1.0 - y_norm).clamp(0, 1)[inside]
        margin_p50 = float(torch.quantile(m, 0.5).item())
        margin_p90 = float(torch.quantile(m, 0.9).item())
    else:
        margin_p50 = float("nan")
        margin_p90 = float("nan")

    fairness = []
    for g_val in gamma_sweep:
        cov_g = float((y_norm <= g_val).float().mean().item())
        ar_proxy = area_ratio * g_val if (area_ratio == area_ratio) else float("nan")
        fairness.append({"gamma": float(g_val), "coverage": cov_g, "area_ratio_proxy": ar_proxy})

    return {
        "coverage": cov,
        "area_mean": area_mean,
        "area_ratio": area_ratio,
        "CPA": cpa,
        "outside_severity": sev,
        "inside_margin_p50": margin_p50,
        "inside_margin_p90": margin_p90,
        "fairness_curve": fairness,
        "n": int(n),
    }


def format_geom_block(title: str, geo: Dict[str, object]) -> str:
    """In đẹp kết quả đánh giá hình học dạng dict."""
    lines = [f"-- GEOMETRIC SIMILARITY: {title} --"]
    lines.append(f"  coverage:          {geo.get('coverage', float('nan')):.4f}")
    if isinstance(geo.get("area_mean"), float):
        lines.append(f"  area_mean:         {geo['area_mean']:.4f}")
    if isinstance(geo.get("area_ratio"), float) and geo["area_ratio"] == geo["area_ratio"]:
        lines.append(f"  area_ratio:        {geo['area_ratio']:.4f}")
        if isinstance(geo.get("CPA"), float) and geo["CPA"] == geo["CPA"]:
            lines.append(f"  CPA:               {geo['CPA']:.4f}")
    lines.append(f"  outside_severity:  {geo.get('outside_severity', float('nan')):.4f}")
    lines.append(f"  inside_margin_p50: {geo.get('inside_margin_p50', float('nan')):.4f}")
    lines.append(f"  inside_margin_p90: {geo.get('inside_margin_p90', float('nan')):.4f}")
    lines.append("  -- fairness sweep --")
    for f in geo.get("fairness_curve", []):
        if f["area_ratio_proxy"] == f["area_ratio_proxy"]:
            lines.append(f"  gamma={f['gamma']:.2f}: cov={f['coverage']:.4f}, AR~{f['area_ratio_proxy']:.4f}")
        else:
            lines.append(f"  gamma={f['gamma']:.2f}: cov={f['coverage']:.4f}")
    return "\n".join(lines)

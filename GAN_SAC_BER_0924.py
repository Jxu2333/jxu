import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import random
import torch.fft as fft
from typing import Tuple, List, Dict, Optional
import os, matplotlib

matplotlib.use("Agg")
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dataclasses import dataclass, field
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from matplotlib.lines import Line2D
from torch.optim.lr_scheduler import StepLR
from math import sqrt


# —— 插入：reward 归一化工具 ——  （放在 import 之后）
class RewardNormalizer:
    def __init__(self, eps: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x: float):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += delta * delta2

    def normalize(self, x: float) -> float:
        std = (self.var / self.count) ** 0.5
        return (x - self.mean) / (std + 1e-8)


# 设置随机种子
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 获取设备
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# 杂波模型类：生成聚类散射杂波协方差矩阵
class ClutterModel:
    def __init__(self, config):
        self.config = config
        self.Nu = config.antenna_num
        self.Nx = int((self.Nu) ** 0.5)
        self.Ny = self.Nx
        self.lambda_ = 3e8 / config.carrier_freq
        self.dx = self.lambda_ / 2
        self.dy = self.lambda_ / 2
        self.J = config.clutter_clusters
        self.a = 2
        self.b = 2
        self.sigma_theta = torch.deg2rad(torch.tensor(config.clutter_angular_spread))
        self.sigma_phi = torch.deg2rad(torch.tensor(config.clutter_angular_spread))

    def compute_clutter_covariance(self, target_pos, uav_pos):
        """计算杂波协方差矩阵（完全GPU版本）"""
        device = uav_pos.device

        # 相对位置与距离
        rel_pos = target_pos - uav_pos[:2]
        distance = torch.norm(torch.cat([rel_pos, -uav_pos[2:3]]))

        theta_q = torch.atan2(rel_pos[1], rel_pos[0])
        phi_q = torch.atan2(torch.norm(rel_pos), uav_pos[2])

        # 散射簇功率分布
        P_j = torch.exp(-torch.arange(self.J, device=device, dtype=torch.float32) / 5)
        P_j = P_j / P_j.sum()

        delta_theta = (torch.rand(self.J, device=device) * 2 - 1) * (torch.pi / 8)
        delta_phi = (torch.rand(self.J, device=device) * 2 - 1) * (torch.pi / 16)
        theta_j = theta_q + delta_theta
        phi_j = phi_q + delta_phi

        A = 1.0
        beta_q = torch.sqrt(self.lambda_ ** 2 / ((4 * torch.pi * distance) ** 2))

        # 使用向量化计算代替双重循环
        R_q = self._compute_R_vectorized(theta_j, phi_j, P_j, A, beta_q)

        return R_q

    def _compute_R_vectorized(self, theta_j, phi_j, P_j, A, beta_q):
        """向量化计算协方差矩阵以提高GPU效率"""
        device = theta_j.device

        # 创建天线索引网格
        kappa_indices = torch.arange(self.Nu, device=device)
        ell_indices = torch.arange(self.Nu, device=device)
        kappa_grid, ell_grid = torch.meshgrid(kappa_indices, ell_indices, indexing='ij')

        # 计算天线位置差
        kappa_x = torch.div(kappa_grid, self.Ny, rounding_mode='floor')
        kappa_y = torch.remainder(kappa_grid, self.Ny)
        ell_x = torch.div(ell_grid, self.Ny, rounding_mode='floor')
        ell_y = torch.remainder(ell_grid, self.Ny)

        d_hat_x = (kappa_x - ell_x).float() * self.dx
        d_hat_y = (kappa_y - ell_y).float() * self.dy

        # 初始化结果矩阵
        R_q = torch.zeros((self.Nu, self.Nu), dtype=torch.cfloat, device=device)

        # 对每个散射簇进行计算
        amplitude_sum = torch.sum(P_j * (torch.sin(theta_j) ** self.a) * (torch.sin(phi_j) ** (self.b + 1)))

        for j in range(self.J):
            B_j = 2 * torch.pi * (
                    d_hat_x * torch.cos(theta_j[j]) * torch.cos(phi_j[j]) +
                    d_hat_y * torch.cos(phi_j[j]) * torch.sin(theta_j[j])
            )
            C_j = 2 * torch.pi * (
                    -d_hat_x * torch.sin(theta_j[j]) * torch.sin(phi_j[j]) +
                    d_hat_y * torch.sin(theta_j[j]) * torch.cos(phi_j[j])
            )

            sigma_eff = self.sigma_theta ** 2 / (1 + C_j * self.sigma_theta ** 2 * self.sigma_phi ** 2)

            phase = torch.exp(1j * 2 * torch.pi * (
                    d_hat_x * torch.sin(theta_j[j]) * torch.cos(phi_j[j]) +
                    d_hat_y * torch.sin(theta_j[j]) * torch.sin(phi_j[j])
            ))

            gaussian = torch.exp(-(B_j ** 2 * sigma_eff + C_j ** 2 * self.sigma_phi ** 2) / 2)

            T_j = (torch.sin(theta_j[j]) ** self.a) * (torch.sin(phi_j[j]) ** (self.b + 1) +
                                                       1j * (self.b + 1) * (torch.sin(phi_j[j]) ** self.b) * torch.cos(
                        phi_j[j]) *
                                                       self.sigma_phi ** 2 * C_j)

            U_j = (self.a * (torch.sin(theta_j[j]) ** (self.a - 1)) * torch.cos(theta_j[j]) *
                   (torch.sin(phi_j[j]) ** (self.b + 1)) +
                   1j * self.sigma_phi ** 2 * (self.b + 1) * (torch.sin(theta_j[j]) ** self.b) *
                   torch.cos(theta_j[j]) * (torch.sin(phi_j[j]) ** self.a) * C_j)

            R_q += P_j[j] * phase * (sigma_eff / self.sigma_theta) * gaussian * (T_j - 1j * B_j * sigma_eff * U_j)

        return A * beta_q * amplitude_sum * R_q


# GLRT检测器类：用于计算SCNR并判决目标检测
class GLRTDetector:
    """广义似然比检验(GLRT)检测器"""

    def __init__(self, config, clutter_model):
        self.config = config
        self.device = torch.device(config.device)
        self.clutter_model = clutter_model

    def estimate_clutter_subspace(self, R_q, rank_threshold=0.9):
        eigenvalues, eigenvectors = torch.linalg.eigh(R_q)
        idx = torch.argsort(eigenvalues, descending=True)
        vals = eigenvalues[idx];
        vecs = eigenvectors[:, idx]
        total_energy = vals.sum()
        cumsum = torch.cumsum(vals, dim=0)
        r_c = torch.argmax(cumsum >= rank_threshold * total_energy) + 1
        return vecs[:, :r_c], r_c

    def project_orthogonal(self, y_s, U_c):
        """将回波信号投影到杂波子空间的正交补上"""
        P_orth = np.eye(U_c.shape[0]) - U_c @ U_c.conj().T
        y_proj = P_orth @ y_s  # 杂波抑制后的信号
        return y_proj, P_orth

    def glrt_detection(self, y_s, R_q, sigma_s_squared):
        Nu = R_q.shape[0]
        MN = y_s.numel() // Nu
        U_c, _ = self.estimate_clutter_subspace(R_q)
        P_c = torch.eye(Nu, device=R_q.device) - U_c @ U_c.conj().T
        Y_mat = y_s.view(MN, Nu)
        Y_proj = Y_mat @ P_c.T
        y_proj = Y_proj.flatten().unsqueeze(1)
        signal_power = torch.norm(y_proj) ** 2
        clutter_power = MN * torch.trace(P_c @ R_q @ P_c.conj().T)
        noise_power = sigma_s_squared * MN * torch.trace(P_c)
        stat = signal_power / (clutter_power + noise_power + 1e-10)
        scnr_proj = signal_power / (clutter_power + noise_power + 1e-10)
        detection = stat > self.config.glrt_threshold
        return detection, stat, scnr_proj


# OTFS信号处理类：生成通信/感知信号和执行延迟-多普勒域变换
class OTFSProcessor:

    def _snr_to_noisevar(self, snr_db: float, es: float = 1.0) -> float:
        snr_linear = 10 ** (snr_db / 10.0)
        sigma2 = es / snr_linear
        return float(sigma2)

    def _constellation_and_labels(self, scheme: str, device):
        from math import sqrt
        import torch
        scheme = scheme.upper()
        if scheme == 'QPSK':
            pts = torch.tensor([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=torch.cfloat, device=device) / sqrt(2)
            labs = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=torch.int8, device=device)
        elif scheme == '16QAM':
            lv = torch.tensor([-3, -1, 1, 3], device=device, dtype=torch.float32)
            inv_gray = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]], device=device, dtype=torch.int8)
            pts_list, labs_list = [], []
            for i in range(4):
                for q in range(4):
                    pts_list.append((lv[i] + 1j * lv[q]) / (10 ** 0.5))
                    labs_list.append(torch.cat([inv_gray[i], inv_gray[q]]))
            pts = torch.stack([torch.tensor(p, dtype=torch.cfloat, device=device) for p in pts_list])
            labs = torch.stack(labs_list)
        else:
            raise ValueError(f"Unsupported scheme: {scheme}")
        return pts, labs

    @torch.no_grad()
    def _llr_maxlog(self, rx_syms, scheme: str, noise_var: float):
        import torch
        pts, labs = self._constellation_and_labels(scheme, rx_syms.device)
        B = labs.shape[1]
        y = rx_syms.view(-1).unsqueeze(1)
        d2 = torch.abs(y - pts.unsqueeze(0)) ** 2
        llrs = []
        for b in range(B):
            mask0 = (labs[:, b] == 0).unsqueeze(0).expand_as(d2)
            mask1 = ~mask0
            m0 = torch.min(d2.masked_fill(~mask0, float('inf')), dim=1).values
            m1 = torch.min(d2.masked_fill(~mask1, float('inf')), dim=1).values
            llr_b = (m1 - m0) / noise_var
            llrs.append(llr_b)
        return torch.stack(llrs, dim=1).reshape(-1)

    def evaluate_ber_coded(self, scheme: str, snr_db_list, frames: int = 200, code: str = 'conv_r12_k7'):
        import torch
        results = {}
        M, N = self.M, self.N
        scheme_u = scheme.upper()
        bits_per_sym = 2 if scheme_u == 'QPSK' else 4 if scheme_u == '16QAM' else None
        if bits_per_sym is None:
            raise ValueError(f"Unsupported scheme: {scheme}")
        S = M * N * bits_per_sym
        for snr_db in snr_db_list:
            total_err, total_bits = 0, 0
            noise_var = self._snr_to_noisevar(snr_db, es=1.0)
            for _ in range(frames):
                if code == 'conv_r12_k7':
                    L_info = max(1, (S // 2) - (ConvCodeR12K7.K - 1))
                    info = torch.randint(0, 2, (L_info,), device=self.device, dtype=torch.int8)
                    coded = ConvCodeR12K7.encode(info)
                    bits_tx = coded[:S] if coded.numel() >= S else torch.cat(
                        [coded, torch.zeros(S - coded.numel(), dtype=torch.int8, device=self.device)], dim=0)
                elif code == 'none':
                    bits_tx = torch.randint(0, 2, (S,), device=self.device, dtype=torch.int8)
                    info = bits_tx
                else:
                    raise ValueError(f"Unknown code: {code}")
                syms = self._modulate(bits_tx.to(torch.int64), scheme_u)
                x_dd = syms.view(N, M).to(torch.cfloat)
                x_tf = self.dd_to_tf(x_dd)
                y_tf_noisy = self.add_noise(x_tf, snr_db) if hasattr(self, 'add_noise') else x_tf + (
                            torch.randn_like(x_tf) + 1j * torch.randn_like(x_tf)) * (noise_var / 2.0) ** 0.5
                y_dd = self.tf_to_dd(y_tf_noisy)
                rx_syms = y_dd.reshape(-1)
                llr = self._llr_maxlog(rx_syms, scheme_u, noise_var)
                if code == 'conv_r12_k7':
                    needed = 2 * (info.numel() + ConvCodeR12K7.K - 1)
                    llr_coded = llr[:needed]
                    info_hat = ConvCodeR12K7.viterbi_decode(llr_coded)
                    L_cmp = int(min(info_hat.numel(), info.numel()))
                    if L_cmp > 0:
                        total_err += int((info_hat[:L_cmp] ^ info[:L_cmp]).sum().item())
                        total_bits += L_cmp
                else:
                    bits_hat = (llr < 0).to(torch.int8)
                    L_cmp = int(min(bits_hat.numel(), bits_tx.numel()))
                    total_err += int((bits_hat[:L_cmp] ^ bits_tx[:L_cmp]).sum().item())
                    total_bits += L_cmp
            results[snr_db] = (total_err / max(1, total_bits))
        return results

    def evaluate_ber_coded_ofdm(self, scheme: str, snr_db_list, frames: int = 200, code: str = 'conv_r12_k7',
                                nfft: int = None, awgn_only: bool = True):
        import torch
        results = {}
        M, N = self.M, self.N
        scheme_u = scheme.upper()
        bits_per_sym = 2 if scheme_u == 'QPSK' else 4 if scheme_u == '16QAM' else None
        if bits_per_sym is None:
            raise ValueError(f"Unsupported scheme: {scheme}")
        S = M * N * bits_per_sym
        Nfft = int(nfft) if nfft is not None else N
        for snr_db in snr_db_list:
            total_err, total_bits = 0, 0
            noise_var = self._snr_to_noisevar(snr_db, es=1.0)
            for _ in range(frames):
                if code == 'conv_r12_k7':
                    L_info = max(1, (S // 2) - (ConvCodeR12K7.K - 1))
                    info = torch.randint(0, 2, (L_info,), device=self.device, dtype=torch.int8)
                    coded = ConvCodeR12K7.encode(info)
                    bits_tx = coded[:S] if coded.numel() >= S else torch.cat(
                        [coded, torch.zeros(S - coded.numel(), dtype=torch.int8, device=self.device)], dim=0)
                elif code == 'none':
                    bits_tx = torch.randint(0, 2, (S,), device=self.device, dtype=torch.int8)
                    info = bits_tx
                else:
                    raise ValueError(f"Unknown code: {code}")
                syms = self._modulate(bits_tx.to(torch.int64), scheme_u)
                X = syms.view(M, N)  # [symbols, subcarriers]
                x_time = torch.fft.ifft(X, n=Nfft, dim=1)
                Y = torch.fft.fft(x_time, n=Nfft, dim=1)[:, :N]
                noise = (torch.randn_like(Y) + 1j * torch.randn_like(Y)) * (noise_var / 2.0) ** 0.5
                Yn = Y + noise
                rx_syms = Yn.reshape(-1)
                llr = self._llr_maxlog(rx_syms, scheme_u, noise_var)
                if code == 'conv_r12_k7':
                    needed = 2 * (info.numel() + ConvCodeR12K7.K - 1)
                    llr_coded = llr[:needed]
                    info_hat = ConvCodeR12K7.viterbi_decode(llr_coded)
                    L_cmp = int(min(info_hat.numel(), info.numel()))
                    if L_cmp > 0:
                        total_err += int((info_hat[:L_cmp] ^ info[:L_cmp]).sum().item())
                        total_bits += L_cmp
                else:
                    bits_hat = (llr < 0).to(torch.int8)
                    L_cmp = int(min(bits_hat.numel(), bits_tx.numel()))
                    total_err += int((bits_hat[:L_cmp] ^ bits_tx[:L_cmp]).sum().item())
                    total_bits += L_cmp
            results[snr_db] = (total_err / max(1, total_bits))
        return results

    """OTFS信号处理器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.M = config.subcarriers
        self.N = config.time_intervals
        self.delta_f = config.bandwidth / self.M
        self.T = 1 / self.delta_f

    def dd_to_tf(self, x_dd: torch.Tensor) -> torch.Tensor:
        if x_dd.dim() == 2:
            x_shifted = fft.ifftshift(x_dd, dim=(-2, -1))
            return fft.ifft2(x_shifted, norm=None) * torch.sqrt(
                torch.tensor(self.M * self.N, device=x_dd.device))
        else:
            x_tf = torch.stack([
                fft.ifft2(fft.ifftshift(x, dim=(-2, -1)), norm=None)
                * torch.sqrt(torch.tensor(self.M * self.N, device=x_dd.device))
                for x in x_dd])
            return x_tf

    def tf_to_dd(self, y_tf: torch.Tensor) -> torch.Tensor:
        if y_tf.dim() == 2:
            X = fft.fft2(y_tf, norm=None)
            return fft.fftshift(X, dim=(-2, -1)) / torch.sqrt(
                torch.tensor(self.M * self.N, device=y_tf.device))
        else:
            y_dd = torch.stack([
                fft.fftshift(fft.fft2(y, norm=None), dim=(-2, -1))
                / torch.sqrt(torch.tensor(self.M * self.N, device=y.device))
                for y in y_tf])
            return y_dd

    def generate_dd_symbols(self, num_users, generate_sensing=False):
        x_comm_dd = torch.zeros(num_users, self.N, self.M, dtype=torch.cfloat, device=self.device)
        for i in range(num_users):
            seq = self._generate_zc_sequence(self.M).to(self.device)
            x_comm_dd[i, 0] = seq
        x_sense_dd = torch.zeros(self.N, self.M, dtype=torch.cfloat, device=self.device)
        if generate_sensing:
            x_sense_dd[0] = self._generate_zc_sequence(self.M).to(self.device)
        return x_comm_dd, x_sense_dd

    def _generate_zc_sequence(self, length):
        u = 1
        n = torch.arange(length, dtype=torch.float32)
        if length % 2 == 0:
            zc = torch.exp(-1j * torch.pi * u * n * (n + 1) / length)
        else:
            zc = torch.exp(-1j * torch.pi * u * n * n / length)
        return zc

    def apply_channel(self, x_tf, h, tau, nu):
        k_dopp = int(torch.round(nu * self.N * self.T).item()) % self.N
        l_delay = int(torch.round(tau * self.M * self.delta_f).item()) % self.M

        if x_tf.dim() == 2:
            y_tf = torch.roll(torch.roll(x_tf, k_dopp, dims=0), l_delay, dims=1) * h
        else:
            # 批量输入
            y_tf = torch.zeros_like(x_tf)
            for i in range(x_tf.size(0)):
                y_tf[i] = torch.roll(torch.roll(x_tf[i], k_dopp, dims=0), l_delay, dims=1) * h[i]
        return y_tf

    def add_noise(self, signal, snr_db):
        power = torch.mean(torch.abs(signal) ** 2)
        noise_p = power / (10 ** (snr_db / 10))
        noise = torch.sqrt(noise_p / 2) * (torch.randn_like(signal) + 1j * torch.randn_like(signal))
        return signal + noise

    # === 新增：调制与解调 ===
    def _modulate(self, bits: torch.Tensor, scheme: str) -> torch.Tensor:
        """将比特映射为复符号，平均功率归一化为1。
        支持: 'QPSK', '16QAM'
        bits: [num_bits] 的0/1张量
        返回: [num_syms] 复符号张量
        """
        scheme = scheme.upper()
        if scheme == 'QPSK':
            assert bits.numel() % 2 == 0, "QPSK需要偶数个比特"
            b = bits.view(-1, 2)
            # Gray: 00->(1+1j), 01->(-1+1j), 11->(-1-1j), 10->(1-1j)
            m = torch.zeros(b.size(0), dtype=torch.cfloat, device=bits.device)
            i0 = (b[:, 0] == 0) & (b[:, 1] == 0)
            i1 = (b[:, 0] == 0) & (b[:, 1] == 1)
            i2 = (b[:, 0] == 1) & (b[:, 1] == 1)
            i3 = (b[:, 0] == 1) & (b[:, 1] == 0)
            m[i0] = 1 + 1j
            m[i1] = -1 + 1j
            m[i2] = -1 - 1j
            m[i3] = 1 - 1j
            m = m / sqrt(2)
            return m
        elif scheme == '16QAM':
            assert bits.numel() % 4 == 0, "16QAM需要4比特每符号"
            b = bits.view(-1, 4)

            # Gray mapping for 16QAM on I and Q: bits (b0 b1) -> levels [-3,-1,1,3]
            def gray_to_level(two_bits: torch.Tensor) -> torch.Tensor:
                # 00->-3, 01->-1, 11->1, 10->3
                l = torch.empty(two_bits.size(0), device=bits.device)
                t0 = (two_bits[:, 0] == 0) & (two_bits[:, 1] == 0)
                t1 = (two_bits[:, 0] == 0) & (two_bits[:, 1] == 1)
                t2 = (two_bits[:, 0] == 1) & (two_bits[:, 1] == 1)
                t3 = (two_bits[:, 0] == 1) & (two_bits[:, 1] == 0)
                l[t0] = -3
                l[t1] = -1
                l[t2] = 1
                l[t3] = 3
                return l

            I = gray_to_level(b[:, 0:2])
            Q = gray_to_level(b[:, 2:4])
            # 归一化平均功率为1（16QAM平均能量 = 10），因此缩放 1/sqrt(10)
            const = (I + 1j * Q) / sqrt(10)
            return const.to(torch.cfloat)
        else:
            raise ValueError(f"不支持的调制方式: {scheme}")

    def _demodulate(self, syms: torch.Tensor, scheme: str) -> torch.Tensor:
        """最小距离硬判决解调，返回0/1比特张量。"""
        scheme = scheme.upper()
        if scheme == 'QPSK':
            # 去归一化等效：直接比较实部/虚部符号即可
            x = syms * sqrt(2)
            bits0 = (x.real < 0).long()  # b0: 0 if I>=0 else 1 (匹配上面的映射)
            bits1 = (x.imag < 0).long() ^ bits0  # 依据映射关系构造
            # 更稳妥按四点最近邻
            # 重新用最近邻保证兼容性
            ref = torch.tensor([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], device=syms.device, dtype=torch.cfloat) / sqrt(2)
            dists = torch.abs(syms.unsqueeze(1) - ref.unsqueeze(0))
            idx = torch.argmin(dists, dim=1)
            # idx: 0->00,1->01,2->11,3->10
            map_bits = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]], device=syms.device, dtype=torch.long)
            bits = map_bits[idx].view(-1)
            return bits
        elif scheme == '16QAM':
            # 最近邻到16QAM星座
            levels = torch.tensor([-3, -1, 1, 3], device=syms.device, dtype=torch.float32)
            norm = sqrt(10)
            I = (syms.real * norm).unsqueeze(1)
            Q = (syms.imag * norm).unsqueeze(1)
            di = torch.abs(I - levels)
            dq = torch.abs(Q - levels)
            idx_i = torch.argmin(di, dim=1)
            idx_q = torch.argmin(dq, dim=1)
            # 反Gray映射: levels index 0..3 -> bits: 0:-3->00, 1:-1->01, 2:1->11, 3:3->10
            inv_gray = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]], device=syms.device, dtype=torch.long)
            bits_i = inv_gray[idx_i]
            bits_q = inv_gray[idx_q]
            bits = torch.cat([bits_i, bits_q], dim=1).view(-1)
            return bits
        else:
            raise ValueError(f"不支持的调制方式: {scheme}")

    def evaluate_ber(self, scheme: str, sinr_db_list: List[float], frames: int = 200) -> Dict[float, float]:
        """在OTFS下评估不同SINR的平均误码率（AWGN/平坦信道）。
        - 每个SINR重复frames次；每帧填满一个N×M DD网格的符号。
        返回: {sinr_db: ber}
        """
        results = {}
        M, N = self.M, self.N
        # 每帧符号数
        num_syms = M * N
        bits_per_sym = 2 if scheme.upper() == 'QPSK' else 4
        for snr_db in sinr_db_list:
            total_bits = 0
            total_err = 0
            for _ in range(frames):
                # 生成随机比特
                bits = torch.randint(0, 2, (num_syms * bits_per_sym,), device=self.device)
                tx_syms = self._modulate(bits, scheme)
                # 映射到DD网格
                x_dd = tx_syms.view(N, M).to(torch.cfloat)
                # DD->TF
                x_tf = self.dd_to_tf(x_dd)
                # 平坦信道 h=1, 无延迟/多普勒
                y_tf = x_tf
                # 加噪声 (SNR≈SINR)
                y_tf_noisy = self.add_noise(y_tf, snr_db)
                # TF->DD
                y_dd = self.tf_to_dd(y_tf_noisy)
                # LLR/判决：直接符号级取均值（因IFFT/FFT为酉变换，符号保持)
                rx_syms = y_dd.view(-1)
                rx_bits = self._demodulate(rx_syms, scheme)
                err = (rx_bits != bits).sum().item()
                total_err += err
                total_bits += bits.numel()
            results[float(snr_db)] = total_err / max(1, total_bits)
        return results


# 系统参数配置数据类
@dataclass
class SystemConfig:
    # === Episode/Budget/Objective ===
    energy_budget_j: float = 1200.0  # UAV energy budget (J)
    reward_mode: str = 'ber'  # 'legacy' or 'ber' (BER-driven objective)
    ber_scheme: str = 'QPSK'  # 'QPSK' or '16QAM'
    ber_coded: bool = False  # if True, apply a crude coding gain offset in BER proxy
    ber_agg: str = 'mean'  # 'mean' or 'max' across users (slot-wise aggregation)
    # 场景参数
    area_size: float = 1200  # 正方形区域边长（米）
    height: float = 100  # UAV飞行高度（米）
    time_slots: int = 70  # 飞行时间步数
    slot_duration: float = 1  # 每个时间步持续时间（秒）
    # UAV运动与通信参数
    v_min: float = 20  # UAV最小速度（m/s）
    v_max: float = 40  # UAV最大速度（m/s）
    a_max: float = 5.0  # UAV最大加速度（m/s^2）
    sensing_range: float = 80  # UAV感知有效范围（米）

    # === Power / Energy Config ===
    tx_power: float = 40  # dBm
    tx_power_min_dbm: float = 20.0  # 最小发射功率 (dBm)  —按硬件再调
    tx_power_max_dbm: float = 40.0  # 最大发射功率 (dBm)  —按硬件再调
    # —— Power guardrails ——
    tx_power_min_dbm_comm: float = 22.0
    tx_power_min_dbm_sense: float = 32.0
    tx_power_slew_db: float = 1.0
    sense_share_floor: float = 0.24
    # --- Deficit-driven comm/sense split (alpha) ---
    alpha_min_frac: float = 0.35  # min fraction to COMM; ensures COMM never starves
    alpha_max_frac: float = 0.90  # max fraction to COMM; ensures SENSE retains budget
    alpha_a0: float = 0.0  # bias for logistic gating
    alpha_a1: float = 4.0  # weight: SINR deficit -> COMM up
    alpha_a2: float = 4.0  # weight: SCNR deficit -> COMM down (SENSE up)
    scnr_release_tau: int = 2  # slots to sustain SCNR-done before releasing SENSE power
    sense_power_floor_frac: float = 0.02  # after release, keep a tiny floor for SENSE

    lambda_power_slew: float = 0.05
    lambda_ignore_inrange: float = 0.2

    # 奖励里能耗惩罚权重（按你的量纲做归一化）
    lambda_e: float = 1.0
    energy_budget: float = 1.0  # 归一化用；若已有可复用现成的，就删这行

    antenna_num: int = 16  # 天线阵元数量（UPA阵列64）
    # OTFS调制参数
    subcarriers: int = 32  # 子载波数量
    time_intervals: int = 16  # 时隙数量
    carrier_freq: float = 3.5e9  # 载波频率（Hz）
    bandwidth: float = 20e6  # 信号带宽（Hz）
    # 任务需求参数
    num_users: int = 3  # 地面用户数量
    num_targets: int = 5  # 地面目标数量
    sinr_threshold: float = 5  # [UNUSED] 通信SINR门限（dB，已移除约束用途）
    scnr_threshold: float = 8  # 感知SCNR门限（dB）
    noise_power: float = -110  # 噪声功率（dBm）
    # 杂波参数
    clutter_clusters: int = 10  # 杂波散射簇数量
    clutter_angular_spread: float = 1.5  # 杂波角度扩散（度）
    glrt_threshold: float = 3.0  # GLRT检测门限
    clutter_rank_threshold: float = 0.85  # 杂波子空间能量阈值
    # 能耗模型参数
    c1: float = 0.008  # 推进能耗模型系数1
    c2: float = 0.03  # 推进能耗模型系数2
    g: float = 9.8  # 重力加速度（用于能耗计算）
    # GAN参数
    latent_dim: int = 128  # 生成器潜在向量维度
    gan_lr: float = 1e-4  # GAN学习率
    gan_beta1: float = 0.5  # Adam优化器beta1
    gan_beta2: float = 0.999  # Adam优化器beta2
    lambda_gp: float = 10  # 梯度惩罚项系数
    lambda_phys: float = 0.3  # 物理约束损失权重
    lambda_task: float = 5.0  # 任务相关损失权重
    n_critic: int = 5  # 每次生成器更新前判别器更新次数
    # SAC参数
    sac_lr: float = 1e-4  # 策略和Q网络学习率
    gamma: float = 0.99  # 奖励折扣因子
    tau: float = 1e-3  # 软更新参数
    alpha_lr: float = 1e-4  # 温度参数学习率
    initial_alpha: float = 0.05  # 初始温度参数α
    alpha_min: float = 0.02  # SAC温度α的最小值（避免过早贪婪）
    # —— SCNR 软约束（在线拉格朗日）——
    scnr_target_rate: float = 0.95
    lambda_scnr_init: float = 0.0
    lambda_scnr_lr: float = 0.5
    sense_reward_scale: float = 30.0  # 可根据需要在 10~100 之间调试
    batch_size: int = 128  # SAC经验批次大小
    buffer_size: int = 50000  # 经验回放缓冲区大小
    # 训练参数
    max_epochs: int = 2  # 最大训练epoch数
    trajectories_per_epoch: int = 8  # 每个epoch生成的轨迹数
    episodes_per_trajectory: int = 6  # 每条轨迹进行的交互episode数
    gradient_steps: int = 10  # 每步环境交互后SAC更新次数

    lambda_rl: float = 0.1  # RL-guidance loss 权重，训练时可调
    rl_warmup_steps = 1000  # rl_warmup_steps
    state_dim: int = 0  # Critic 的 state 输入维度
    traj_horizon = time_slots
    traj_point_dim = 2

    # 设备配置
    device: str = field(default_factory=get_device)


# UAV-ISAC环境类：模拟无人机运动、通信和感知交互
class UAVISACEnvironment:

    # === BER estimation helpers (analytic proxy) ===
    def _qfunc(self, x):
        import torch
        return 0.5 * torch.special.erfc(x / (2 ** 0.5))

    def _ber_uncoded(self, sinr_lin: 'torch.Tensor', scheme: str) -> 'torch.Tensor':
        import torch, math
        scheme = scheme.upper()
        g = sinr_lin.clamp(min=1e-12)
        if scheme == 'QPSK':
            return self._qfunc(torch.sqrt(2.0 * g))
        elif scheme == '16QAM':
            M = 16.0
            k = math.log2(M)
            coeff = 4.0 / k * (1.0 - 1.0 / M ** 0.5)
            return coeff * self._qfunc((3.0 * k / (M - 1.0) * g).sqrt())
        else:
            raise ValueError(f"Unknown scheme for BER: {scheme}")

    def _ber_pred(self, sinr_lin: 'torch.Tensor') -> 'torch.Tensor':
        import torch
        scheme = getattr(self.config, 'ber_scheme', 'QPSK')
        coded = bool(getattr(self.config, 'ber_coded', False))
        if coded:
            # crude ~3 dB coding gain (×2 in linear) before computing BER
            return self._ber_uncoded(sinr_lin * 2.0, scheme).clamp(min=1e-6, max=0.5)
        else:
            return self._ber_uncoded(sinr_lin, scheme).clamp(min=1e-6, max=0.5)

    def __init__(self, config: SystemConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.otfs = OTFSProcessor(config)
        self.clutter_model = ClutterModel(config)
        self.glrt_detector = GLRTDetector(config, self.clutter_model)
        self.lambda_scnr = 0.0
        self.reset()

    def reset(self):
        self.user_positions = torch.tensor([[400, 800], [600, 600], [800, 800]],
                                           dtype=torch.float32, device=self.device)

        self.target_positions = torch.tensor([[200, 800], [400, 600], [600, 400], [800, 600], [1000, 800]],
                                             dtype=torch.float32, device=self.device)
        self.uav_position = torch.tensor([600.0, 1000.0, self.config.height],
                                         dtype=torch.float32, device=self.device)
        self.uav_velocity = torch.tensor([0.0, -25.0, 0.0],
                                         dtype=torch.float32, device=self.device)
        self.current_slot = 0
        self.scnr_done_streak = 0  # consecutive slots with all targets SCNR satisfied
        self.prev_tx_dbm = None
        self.sensing_status = torch.zeros(self.config.num_targets,
                                          dtype=torch.float32, device=self.device)
        self.cumulative_scnr = torch.zeros(self.config.num_targets,
                                           dtype=torch.float32, device=self.device)
        self.cumulative_sinr = torch.zeros(self.config.num_users,
                                           dtype=torch.float32, device=self.device)
        self.total_energy = torch.tensor(0.0,
                                         dtype=torch.float32, device=self.device)

        self.x_comm_dd, self.x_sense_dd = self.otfs.generate_dd_symbols(
            self.config.num_users, self.config.num_targets)
        return self._get_state()

    def set_lambda_scnr(self, val: float):
        self.lambda_scnr = float(val)

    def _get_state(self):
        state = []
        state.append(self.uav_position[:2] / self.config.area_size)
        state.append(self.uav_velocity[:2] / self.config.v_max)
        for user_pos in self.user_positions:
            rel_pos = (user_pos - self.uav_position[:2]) / self.config.area_size
            state.extend(rel_pos)
        for target_pos in self.target_positions:
            rel_pos = (target_pos - self.uav_position[:2]) / self.config.area_size
            state.extend(rel_pos)

        # 感知状态
        state.extend(self.sensing_status)
        scnr_threshold_linear = 10 ** (self.config.scnr_threshold / 10)
        normalized_scnr = torch.clamp(self.cumulative_scnr / scnr_threshold_linear, 0.0, 2.0)
        state.extend(normalized_scnr)

        # （已移除SINR阈值相关状态）
        time_progress = torch.tensor([self.current_slot / self.config.time_slots],
                                     device=self.device, dtype=torch.float32)
        energy_normalized = torch.tensor([self.total_energy / 1000.0],
                                         device=self.device, dtype=torch.float32)
        state.append(time_progress)
        state.append(energy_normalized)
        # 返回状态向量
        state_array = torch.cat([part.flatten() for part in state])
        state_array = torch.nan_to_num(state_array, nan=0.0, posinf=2.0, neginf=0.0)
        return state_array

    def _calculate_state_dim(self):
        dim = 4  # UAV位置(x,y,z)和速度(vx,vy,vz)占6维
        dim += 2 * self.config.num_users  # 每个用户相对位置(x,y)
        dim += 2 * self.config.num_targets  # 每个目标相对位置(x,y)
        dim += self.config.num_targets  # 感知状态（目标是否探测到）
        dim += self.config.num_targets  # 累积SCNR（归一化）
        dim += 2  # 时间进度、能耗
        return dim

    def step(self, action: Dict[str, torch.Tensor], trajectory: Optional[torch.Tensor] = None):
        """执行一步动作，更新环境状态"""
        txw = float(action.get('tx_power_w', 10 ** ((self.config.tx_power_min_dbm - 30.) / 10.)))
        power_allocation = action['power']  # 发射功率分配（包含通信和感知功率）
        sensing_decision = action['sensing']  # 感知决策（哪个目标进行感知）
        # 轨迹转换
        if trajectory is not None and not isinstance(trajectory, torch.Tensor):
            trajectory = torch.tensor(trajectory, dtype=torch.float32, device=self.device)
        if trajectory is not None and self.current_slot < trajectory.shape[0]:
            old_position = self.uav_position[:2].clone()
            self.uav_position[:2] = trajectory[self.current_slot]
            if self.current_slot > 0:
                self.uav_velocity[:2] = (self.uav_position[:2] - old_position) / self.config.slot_duration
                self.uav_velocity[2] = 0
        num_targets = self.config.num_targets if action['sensing'].sum() > 0 else 0
        self.x_comm_dd, self.x_sense_dd = self.otfs.generate_dd_symbols(self.config.num_users, num_targets)
        sinr_values = self._calculate_sinr(power_allocation, txw)
        scnr_values = self._calculate_scnr(power_allocation, sensing_decision, txw)
        for q in range(self.config.num_targets):
            if self.sensing_status[q] == 0:
                horizontal_dist_q = torch.norm(self.uav_position[:2] - self.target_positions[q], p=2)
                if horizontal_dist_q <= self.config.sensing_range and sensing_decision[q] > 0:
                    self.cumulative_scnr[q] += scnr_values[q]
                    scnr_threshold_linear = 10 ** (self.config.scnr_threshold / 10)
                    if self.cumulative_scnr[q] >= scnr_threshold_linear:
                        self.sensing_status[q] = 1.0
                        break

        self.cumulative_sinr += sinr_values
        energy = self._calculate_energy(power_allocation, sensing_decision, txw)
        self.total_energy += energy
        reward = self._calculate_reward(sinr_values, scnr_values, energy, power_allocation, sensing_decision)
        tx_dbm = 10 * torch.log10(torch.tensor(txw, device=self.device) * 1e3 + 1e-12)
        if self.prev_tx_dbm is not None:
            reward = reward - self.config.lambda_power_slew * torch.abs(tx_dbm - self.prev_tx_dbm).item()
        self.prev_tx_dbm = tx_dbm.detach()
        scnr_thr_lin = 10 ** (self.config.scnr_threshold / 10)
        debt_ratios = []
        for q in range(self.config.num_targets):
            if self.sensing_status[q] == 0:
                dist_q = torch.norm(self.uav_position[:2] - self.target_positions[q])
                if dist_q <= self.config.sensing_range and sensing_decision[q] <= 0.0:
                    debt = max(0.0, (scnr_thr_lin - self.cumulative_scnr[q]).item())
                    debt_ratios.append(debt / scnr_thr_lin)
        if len(debt_ratios) > 0:
            reward = reward + (- self.config.lambda_ignore_inrange * max(debt_ratios))
        if hasattr(self, 'lambda_scnr'):
            debt_total = torch.clamp((scnr_thr_lin - self.cumulative_scnr) / scnr_thr_lin, min=0.0).sum().item()
            reward = reward + (- float(self.lambda_scnr) * 0.2 * debt_total)

        scnr_thr_lin = 10 ** (self.config.scnr_threshold / 10)

        avg_sinr_now = self.cumulative_sinr / (self.current_slot + 1)

        all_targets_done = torch.all(self.cumulative_scnr >= scnr_thr_lin)
        all_users_done = torch.tensor(False, device=self.device)  # SINR阈值相关已移除
        early_done = bool(all_targets_done.item())

        if all_targets_done:
            self.scnr_done_streak = int(getattr(self, 'scnr_done_streak', 0)) + 1
        else:
            self.scnr_done_streak = 0

        # 更新时间步
        self.current_slot += 1

        done = early_done or (self.current_slot >= self.config.time_slots) or (
                    self.total_energy >= getattr(self.config, 'energy_budget_j', float('inf'))) or (
                           self.total_energy >= getattr(self.config, 'energy_budget_j', float('inf')))

        # 获取下一状态
        next_state = self._get_state()

        # 额外信息（新增 slots_used）
        info = {
            'sinr': sinr_values,
            'scnr': scnr_values,
            'energy': energy,
            'tx_power_w': txw,
            'cumulative_scnr': self.cumulative_scnr.clone(),
            'cumulative_sinr': self.cumulative_sinr.clone(),
            'average_sinr': self.cumulative_sinr / max(self.current_slot, 1),
            'sensing_status': self.sensing_status.clone(),
            'slots_used': int(self.current_slot),  # <— 新增：本次已用时隙数
            'early_done': early_done,
            'energy_total_j': float(self.total_energy.item()),
            'energy_budget_exceeded': bool(self.total_energy >= getattr(self.config, 'energy_budget_j', float('inf'))),
            'all_targets_done': bool(all_targets_done.item()),
            'all_users_done': bool(all_users_done.item()),
            'scnr_done_streak': int(self.scnr_done_streak),
        }
        return next_state, reward, done, info

    def _calculate_sinr(self, power_allocation, tx_power_watts):
        """计算SINR（使用传入的总发射功率，单位：W）"""
        device = self.device
        if not isinstance(power_allocation, torch.Tensor):
            power_allocation = torch.tensor(power_allocation, dtype=torch.float32, device=device)

        sinr_values = torch.zeros(self.config.num_users, device=self.device)
        tx_power_watts = float(tx_power_watts)  # 确保是标量

        for p in range(self.config.num_users):
            rel_pos = self.user_positions[p] - self.uav_position[:2]
            distance = torch.norm(torch.cat([rel_pos, torch.tensor([0.], device=self.device)]))

            azimuth = torch.atan2(rel_pos[1], rel_pos[0])
            height_tensor = torch.tensor(self.config.height, device=self.device, dtype=rel_pos.dtype)
            elevation = torch.atan2(height_tensor, torch.norm(rel_pos))

            a_t = self._compute_steering_vector(azimuth, elevation)
            a_t = a_t / (torch.linalg.norm(a_t) + 1e-12)

            wavelength = 3e8 / self.config.carrier_freq
            path_loss = (wavelength / (4 * np.pi * distance)) ** 2

            if distance > 1e-6:
                velocity_los = torch.dot(self.uav_velocity[:2],
                                         (self.user_positions[p] - self.uav_position[:2])) / distance
            else:
                velocity_los = 0
            doppler_shift = velocity_los * self.config.carrier_freq / 3e8
            nu_norm = doppler_shift * self.otfs.N * self.otfs.T
            tau_norm = (distance / 3e8) * self.otfs.M * self.otfs.delta_f

            if 1 + p < len(power_allocation):
                w_p = torch.sqrt(power_allocation[1 + p] * tx_power_watts / (self.otfs.N * self.otfs.T)) * a_t
            else:
                avg_power = (1.0 / (self.config.num_users + 1))
                w_p = torch.sqrt(avg_power * tx_power_watts / (self.otfs.N * self.otfs.T)) * a_t

            e_hp = torch.sqrt(path_loss)
            alpha_p = e_hp * torch.vdot(a_t, w_p)

            x_p_tf = self.otfs.dd_to_tf(self.x_comm_dd[p])
            h_p = alpha_p
            y_p_tf = self.otfs.apply_channel(x_p_tf, h_p, tau_norm, nu_norm)

            # 干扰
            interference_tf = torch.zeros_like(y_p_tf)
            for i in range(self.config.num_users):
                if i == p:
                    continue
                user_i_pos_3d = torch.cat([self.user_positions[i], torch.tensor([0.0], device=self.device)])
                rel_pos_i = user_i_pos_3d - self.uav_position
                distance_i = torch.norm(rel_pos_i)
                path_loss_i = (wavelength / (4 * torch.pi * distance_i)) ** 2
                rel_pos_2d_i = self.user_positions[i] - self.uav_position[:2]
                azimuth_i = torch.atan2(rel_pos_2d_i[1], rel_pos_2d_i[0])
                elevation_i = torch.atan2(height_tensor, torch.norm(rel_pos_2d_i))
                a_t_i = self._compute_steering_vector(azimuth_i, elevation_i)
                a_t_i = a_t_i / (torch.linalg.norm(a_t_i) + 1e-12)

                if 1 + i < len(power_allocation):
                    power_value = power_allocation[1 + i] * tx_power_watts / (self.otfs.N * self.otfs.T)
                    w_i = torch.sqrt(power_value.clone().detach()) * a_t_i
                else:
                    avg_power = (1.0 / (self.config.num_users + 1))
                    power_value = avg_power * tx_power_watts / (self.otfs.N * self.otfs.T)
                    w_i = torch.sqrt(power_value.clone().detach()) * a_t_i

                e_hi = torch.sqrt(path_loss_i)
                alpha_i = e_hi * torch.vdot(a_t, w_i)

                if distance_i > 1e-6:
                    velocity_los_i = torch.dot(self.uav_velocity[:2],
                                               (self.user_positions[i] - self.uav_position[:2])) / distance_i
                else:
                    velocity_los_i = 0
                doppler_shift_i = velocity_los_i * self.config.carrier_freq / 3e8
                nu_norm_i = doppler_shift_i * self.otfs.N * self.otfs.T
                tau_norm_i = (distance_i / 3e8) * self.otfs.M * self.otfs.delta_f

                x_i_tf = self.otfs.dd_to_tf(self.x_comm_dd[i])
                interference_tf += self.otfs.apply_channel(x_i_tf, alpha_i * 0.1, tau_norm_i, nu_norm_i)

            noise_power_linear = 10 ** (self.config.noise_power / 10) / 1000
            y_p_dd = self.otfs.tf_to_dd(y_p_tf)
            interference_dd = self.otfs.tf_to_dd(interference_tf)

            signal_power = torch.mean(torch.abs(y_p_dd) ** 2)
            interference_power = torch.mean(torch.abs(interference_dd) ** 2)
            sinr_linear = signal_power / (interference_power + self.otfs.M * self.otfs.N * noise_power_linear + 1e-10)
            sinr_values[p] = torch.clamp(sinr_linear, min=1e-6)

        return sinr_values

    def _calculate_scnr(self, power_allocation, sensing_decision, tx_power_watts):
        """计算每个目标的SCNR（使用传入的总发射功率，单位：W）"""
        scnr_values = torch.zeros(self.config.num_targets, device=self.device)
        noise_power_linear = 10 ** (self.config.noise_power / 10) / 1000
        tx_power_watts = float(tx_power_watts)

        for q in range(self.config.num_targets):
            if self.sensing_status[q] == 1:
                continue
            if sensing_decision[q] > 0:
                horizontal_distance = torch.linalg.norm(self.uav_position[:2] - self.target_positions[q])
                target_pos_3d = torch.cat(
                    [self.target_positions[q].clone().detach(), torch.tensor([0.0], device=self.device)])
                distance_3d = torch.norm(self.uav_position - target_pos_3d)
                if horizontal_distance <= self.config.sensing_range:
                    R_q = self.clutter_model.compute_clutter_covariance(self.target_positions[q], self.uav_position)
                    x_0 = self.x_sense_dd.flatten()

                    rel_pos = target_pos_3d - self.uav_position
                    if torch.norm(rel_pos) > 1e-6:
                        line_of_sight = rel_pos / torch.norm(rel_pos)
                        relative_velocity = 2 * torch.dot(self.uav_velocity, line_of_sight)  # 双程
                    else:
                        relative_velocity = torch.tensor(0.0, device=self.device)

                    tau_s = 2 * distance_3d / 3e8
                    nu_s = 2 * relative_velocity * self.config.carrier_freq / 3e8
                    k_q = int(torch.round(nu_s * self.otfs.N * self.otfs.T).item()) % self.otfs.N
                    l_q = int(torch.round(tau_s * self.otfs.M * self.otfs.delta_f).item()) % self.otfs.M

                    rel_pos_2d = self.target_positions[q] - self.uav_position[:2]
                    azimuth = torch.atan2(rel_pos_2d[1], rel_pos_2d[0])
                    elevation = torch.atan2(self.uav_position[2], torch.norm(rel_pos_2d))
                    b_t = self._compute_steering_vector(azimuth, elevation)
                    b_r = b_t

                    p0 = power_allocation[0] if isinstance(power_allocation, torch.Tensor) else torch.tensor(
                        power_allocation[0], device=self.device)
                    u_q = torch.sqrt(p0 * tx_power_watts / self.config.antenna_num) * b_t

                    wavelength = 3e8 / self.config.carrier_freq
                    sigma_rcs = 1.0
                    beta_q_val = torch.sqrt(wavelength ** 2 * sigma_rcs / ((4 * torch.pi) ** 3 * distance_3d ** 4))

                    spatial_scalar = beta_q_val * torch.vdot(b_t, u_q)
                    H_dd = self._construct_dd_channel_matrix(k_q, l_q)
                    s_dd = H_dd @ x_0

                    signal_power = (torch.abs(spatial_scalar) ** 2) * torch.norm(b_r) ** 2 * torch.norm(s_dd) ** 2
                    clutter_power = torch.trace(R_q).real
                    total_noise_power = self.otfs.M * self.otfs.N * self.config.antenna_num * noise_power_linear

                    scnr_values[q] = signal_power / (clutter_power + total_noise_power + 1e-16)
                else:
                    scnr_values[q] = 0
            else:
                scnr_values[q] = 0
        return scnr_values

    def _compute_steering_vector(self, azimuth, elevation):
        """计算UPA阵列的导向向量"""
        wavelength = 3e8 / self.config.carrier_freq
        dx = wavelength / 2
        dy = wavelength / 2
        steering_vector = []
        # 逐元素计算阵列响应
        for nx in range(self.clutter_model.Nx):
            for ny in range(self.clutter_model.Ny):
                phase = 2 * torch.pi / wavelength * (
                        nx * dx * torch.sin(elevation) * torch.cos(azimuth) +
                        ny * dy * torch.sin(elevation) * torch.sin(azimuth)
                )
                steering_vector.append(torch.exp(1j * phase))
        return torch.stack(steering_vector)

    def _construct_dd_channel_matrix(self, k_q, l_q):
        """构建延迟-多普勒域感知信道矩阵"""
        H_dd = torch.zeros((self.otfs.M * self.otfs.N, self.otfs.M * self.otfs.N), dtype=torch.cfloat,
                           device=self.device)
        # 单径延迟-多普勒响应矩阵（单位矩阵循环移位）
        for k in range(self.otfs.N):
            for l in range(self.otfs.M):
                idx_in = k * self.otfs.M + l
                k_out = (k + k_q) % self.otfs.N
                l_out = (l + l_q) % self.otfs.M
                idx_out = k_out * self.otfs.M + l_out
                H_dd[idx_out, idx_in] = 1.0
        return H_dd

    def _calculate_energy(self, power_allocation, sensing_decision, tx_power_watts):
        """当前时隙能耗 = 发射能耗(通信+感知) + 推进能耗；tx_power_watts 为总功率(W)"""
        tx_power_watts = torch.as_tensor(tx_power_watts, device=self.device, dtype=torch.float32)

        # 通信能耗：通信占比 * P_t * Δt
        comm_energy = torch.as_tensor(sum(power_allocation[1:]), device=self.device, dtype=torch.float32) \
                      * tx_power_watts * self.config.slot_duration
        # 感知能耗：感知占比 *（是否在感知） * P_t * Δt
        sense_energy = torch.as_tensor(power_allocation[0], device=self.device, dtype=torch.float32) \
                       * torch.as_tensor(sum(sensing_decision), device=self.device, dtype=torch.float32) \
                       * tx_power_watts * self.config.slot_duration

        # 推进能耗：与你原有模型一致
        v_norm = torch.linalg.norm(self.uav_velocity[:2])
        v_norm = max(v_norm, self.config.v_min)
        propulsion_energy = self.config.slot_duration * (self.config.c1 * v_norm ** 2 + self.config.c2 * v_norm + 0.5)

        return comm_energy + sense_energy + propulsion_energy

    def _calculate_reward(self, sinr_values, scnr_values, energy, power_allocation, sensing_decision):
        """根据当前SINR/SCNR/能耗计算即时奖励（GPU版本）"""
        if not isinstance(sinr_values, torch.Tensor):
            sinr_values = torch.tensor(sinr_values, device=self.device)
        if not isinstance(scnr_values, torch.Tensor):
            scnr_values = torch.tensor(scnr_values, device=self.device)
        # === BER-driven objective (slot-wise), averaged across users ===
        if getattr(self.config, 'reward_mode', 'legacy') == 'ber':
            sinr_tensor = sinr_values if isinstance(sinr_values, torch.Tensor) else torch.tensor(sinr_values,
                                                                                                 device=self.device,
                                                                                                 dtype=torch.float32)
            ber_per_user = self._ber_pred(sinr_tensor)  # [U]
            if getattr(self.config, 'ber_agg', 'mean') == 'max':
                ber_obj = torch.max(ber_per_user)
            else:
                ber_obj = torch.mean(ber_per_user)
            r_comm = (-ber_obj).item()
        else:
            # 传统模式禁用SINR门限评价
            r_comm = 0.0

        # 感知奖励
        r_sense = 0.0
        scnr_threshold_linear = 10 ** (self.config.scnr_threshold / 10)
        for q in range(self.config.num_targets):
            if sensing_decision[q] > 0:
                target_pos = self.target_positions[q]
                dist = torch.norm(self.uav_position[:2] - target_pos)

                if dist <= self.config.sensing_range:
                    scnr_ratio = scnr_values[q] / (scnr_threshold_linear + 1e-10)
                    alpha = 3.0
                    r_sense += (1.0 - torch.exp(-alpha * scnr_ratio)).item()

        if sum(sensing_decision) > 0:
            r_sense = (r_sense / sum(sensing_decision)) * self.config.sense_reward_scale
        else:
            r_sense = 0.0

        # 目标覆盖奖励
        coverage_bonus = 0.5 * torch.sum(self.sensing_status).item() / self.config.num_targets
        r_sense = r_sense + coverage_bonus

        # 能耗惩罚
        max_energy_per_slot = 50
        r_energy = -min(energy / max_energy_per_slot, 1.0) * 0.5

        # 约束惩罚
        r_constraint = 0.0
        power_sum = sum(power_allocation)
        if abs(power_sum - 1.0) > 0.01:
            r_constraint -= min(abs(power_sum - 1.0), 1.0) * 0.5
        if sum(sensing_decision) > 1:
            r_constraint -= 0.5

        # 目标距离惩罚（GPU版本）
        r_distance = 0.0
        target_distances = []
        for q in range(self.config.num_targets):
            if self.sensing_status[q] == 0:
                dist = torch.norm(self.uav_position[:2] - self.target_positions[q])
                target_distances.append(torch.clamp(dist / self.config.sensing_range, max=1.0))

        if target_distances:
            # 使用torch.stack和torch.mean替代np.mean
            target_distances_tensor = torch.stack(target_distances)
            r_distance -= 0.5 * torch.mean(target_distances_tensor).item()

        # 总奖励

        # Dynamic sense-reward gating: after SCNR satisfied for tau slots, drop sense weight
        if getattr(self, 'scnr_done_streak', 0) >= int(self.config.scnr_release_tau):
            w_comm = 0.40
            w_sense = 0.05  # de-emphasize sensing once finished
        else:
            w_comm = 0.40
            w_sense = 0.40

        reward = (
                w_comm * r_comm +
                w_sense * r_sense +
                0.10 * r_energy +
                0.05 * r_constraint +
                0.05 * r_distance
        )
        # reward = max(min(reward, 10.0), -2.0)
        return reward


# GAN-SAC联合算法主体
class GANSAC:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.current_epoch = 0
        # 添加平滑奖励跟踪
        self.reward_smooth_alpha = 0.1  # EMA 平滑系数
        self._ema_episode_reward = None  # 平滑奖励状态
        # —— 1. 初始化 Critic (Q) 网络 ——
        #    假设你希望 Critic 直接输入展平轨迹
        traj_flat_dim = config.traj_horizon * config.traj_point_dim
        # 如果你还是想用原来的 44 维，那用 config.action_dim
        action_dim_for_q = traj_flat_dim
        self.q1 = QNetwork(state_dim=config.state_dim,
                           action_dim=action_dim_for_q).to(self.device)
        self.q2 = QNetwork(state_dim=config.state_dim,
                           action_dim=action_dim_for_q).to(self.device)
        self.env = UAVISACEnvironment(config)
        self.prev_total_dbm = None
        self.lambda_scnr = self.config.lambda_scnr_init
        self.env.set_lambda_scnr(self.lambda_scnr)
        self.state_dim = self.env._calculate_state_dim()
        self.action_dim = config.num_users + config.num_targets + 2  # 动作向量维度（功率+感知）
        # 初始化GAN网络
        self.generator = TrajectoryGenerator(config).to(self.device)
        self.discriminator = TrajectoryDiscriminator(config).to(self.device)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=config.gan_lr,
                                      betas=(config.gan_beta1, config.gan_beta2))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=config.gan_lr,
                                      betas=(config.gan_beta1, config.gan_beta2))
        # 初始化SAC网络
        self.policy = GaussianPolicy(self.state_dim, self.action_dim).to(self.device)
        self.q1 = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.q2 = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.q1_target = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.q2_target = QNetwork(self.state_dim, self.action_dim).to(self.device)
        # —— 2. 在 Critic 初始化后，再拿到 fc1 的 in_features ——
        critic_in = self.q1.fc1.in_features  # 这时就不会报错了

        # —— 3. 定义 traj_encoder，把 335 维映射到 critic_in ——
        self.traj_encoder = nn.Linear(traj_flat_dim, critic_in).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        # 初始化优化器
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.sac_lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=config.sac_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=config.sac_lr)
        # 自适应熵温度参数α
        self.log_alpha = torch.tensor(np.log(config.initial_alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.policy_scheduler = StepLR(self.policy_optimizer, step_size=self.config.max_epochs, gamma=1.0)
        self.q1_scheduler = StepLR(self.q1_optimizer, step_size=self.config.max_epochs, gamma=1.0)
        self.q2_scheduler = StepLR(self.q2_optimizer, step_size=self.config.max_epochs, gamma=1.0)
        self.alpha_scheduler = StepLR(self.alpha_optimizer, step_size=self.config.max_epochs, gamma=1.0)
        self.target_entropy = -self.action_dim * 0.5
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(self.config.buffer_size, self.device)
        # 1) mini-batch & 多步更新
        self.batch_size = 512
        self.gradient_steps = 3
        # 2) ε-greedy 探索
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        # 在 max_epochs 内线性衰减到 min_epsilon
        self.epsilon_decay = (self.epsilon - self.min_epsilon) / self.config.max_epochs
        # 3) 奖励归一化
        self.reward_normalizer = RewardNormalizer()
        # 训练统计字典
        self.training_stats = {
            'g_loss': [],
            'g_loss_rl': [],
            'd_loss': [],
            'q_loss': [],
            'policy_loss': [],
            'alpha_loss': [],
            'rewards': [],
            'trajectory_quality': [],
            'sinr_satisfaction': [],
            'scnr_satisfaction': [],
            'energy_consumption': [],
            'coverage_rate': [],
            'lambda_scnr': [],
            'episodes': []
        }

    def _log_episode_row(self, epoch: int, traj_idx: int, episode_idx: int, episode_reward, info):
        """
        汇总并记录单个 episode 的关键指标到 self.training_stats['episodes']。
        """
        # 处理 episode_reward
        if hasattr(episode_reward, 'item'):
            ep_r = float(episode_reward.item())
        else:
            ep_r = float(episode_reward)

        # 计算 EMA 平滑奖励
        if self._ema_episode_reward is None:
            self._ema_episode_reward = ep_r
        else:
            alpha = self.reward_smooth_alpha
            self._ema_episode_reward = alpha * ep_r + (1.0 - alpha) * self._ema_episode_reward

                # SINR阈值相关已移除，仅保留SCNR阈值
        scnr_thr_lin = 10 ** (self.config.scnr_threshold / 10)
        avg_sinr = info['average_sinr']  # tensor[users]
        cum_scnr = info['cumulative_scnr']  # tensor[targets]
        sensing_status = info['sensing_status']  # tensor[targets]
        sinr_sat_rate = 0.0  # 占位：不再计算SINR满足率
        scnr_sat_rate = (cum_scnr >= scnr_thr_lin).float().mean().item()
        coverage_rate = sensing_status.float().mean().item()

        row = {
            'epoch': int(epoch),
            'trajectory_idx': int(traj_idx),
            'episode_idx': int(episode_idx),
            'episode_reward': ep_r,
            'episode_reward_ema': float(self._ema_episode_reward),  # 新增：平滑奖励
            'slots_used': int(info.get('slots_used', self.config.time_slots)),
            'early_done': bool(info.get('early_done', False)),
            'all_targets_done': bool(info.get('all_targets_done', False)),
            'all_users_done': bool(info.get('all_users_done', False)),
            'sinr_sat_rate': float(sinr_sat_rate),
            'scnr_sat_rate': float(scnr_sat_rate),
            'coverage_rate': float(coverage_rate),
            'energy_total': float(self.env.total_energy.item())
        }

        self.training_stats['episodes'].append(row)

    def export_episodes_csv(self, path: str = 'episodes.csv'):
        """
        将 self.training_stats['episodes'] 导出为 CSV。
        """
        rows = self.training_stats.get('episodes', [])
        if not rows:
            print("[export_episodes_csv] 没有可导出的 episode 数据。")
            return

        # 统一字段顺序，包含平滑奖励
        fieldnames = [
            'epoch', 'trajectory_idx', 'episode_idx',
            'episode_reward', 'episode_reward_ema',  # 包含平滑奖励
            'slots_used', 'early_done',
            'all_targets_done', 'all_users_done',
            'sinr_sat_rate', 'scnr_sat_rate', 'coverage_rate',
            'energy_total'
        ]

        # 兜底：如有额外字段，追加到末尾
        extra = [k for k in rows[0].keys() if k not in fieldnames]
        fieldnames += extra

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        print(f"[export_episodes_csv] 已导出 {len(rows)} 条记录到 {path}")
        print(f"包含字段: {', '.join(fieldnames)}")

    def train(self):
        """训练GAN-SAC"""

        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            # Curriculum learning: 逐渐增加任务难度
            difficulty_factor = min(1.0, (epoch + 1) / 30)  # 前20个epoch逐渐增加难度
            self.reward_weights = {
                'comm': 0.25 + 0.05 * difficulty_factor,
                'sense': 0.20 + 0.10 * difficulty_factor,
                'energy': 0.15 + 0.05 * difficulty_factor,
                'constraint': 0.05,
                'exploration': 0.25 - 0.10 * difficulty_factor,  # 早期更多探索
                'time_bonus': 0.05 * difficulty_factor
            }

            # 阶段1：生成轨迹
            trajectories = self._generate_trajectories()

            # 阶段2：SAC资源分配优化
            trajectory_rewards = []
            trajectory_sinr_sat = []
            trajectory_scnr_sat = []

            for traj_idx, trajectory in enumerate(trajectories):
                rewards = self._optimize_resource_allocation(trajectory, traj_idx=traj_idx)
                # 1) 把列表转换成 Tensor
                rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
                # 2) 计算均值并变成 Python float
                avg_reward = rewards_tensor.mean().item()
                trajectory_rewards.append(avg_reward)

                # 评估该轨迹的性能
                state = self.env.reset()
                sinr_satisfied = []

                for t in range(self.config.time_slots):
                    action = self._select_action(state)
                    next_state, reward, done, info = self.env.step(action, trajectory)

                    # 记录SINR满足情况
                    sinr_sat = 0.0  # SINR阈值相关已移除
                    sinr_satisfied.append(sinr_sat)

                    state = next_state
                    if done:
                        break

                # 记录SCNR满足情况
                scnr_threshold_linear = 10 ** (self.config.scnr_threshold / 10)
                scnr_bool = (info['cumulative_scnr'] >= scnr_threshold_linear)  # torch.BoolTensor on cuda
                scnr_sat = scnr_bool.float().mean().item()

                trajectory_sinr_sat.append(np.mean(sinr_satisfied))
                trajectory_scnr_sat.append(scnr_sat)

            # 记录统计信息
            self.training_stats['sinr_satisfaction'].append(np.mean(trajectory_sinr_sat))
            self.training_stats['scnr_satisfaction'].append(np.mean(trajectory_scnr_sat))

            # —— Online update of SCNR Lagrange multiplier ——
            scnr_rate = self.training_stats['scnr_satisfaction'][-1]
            self.lambda_scnr = max(0.0, float(getattr(self, 'lambda_scnr', 0.0)) + self.config.lambda_scnr_lr * (
                        self.config.scnr_target_rate - scnr_rate))
            self.env.set_lambda_scnr(self.lambda_scnr)
            self.training_stats.setdefault('lambda_scnr', []).append(self.lambda_scnr)

            # 阶段3：更新GAN
            self._update_gan(trajectories, trajectory_rewards)

            # 阶段4：记录统计信息
            if epoch % 1 == 0:  # 每个epoch都记录
                self._log_training_stats(epoch)

    def _generate_trajectories(self):
        """利用生成器生成一定数量的无人机轨迹"""
        trajectories = []
        task_params = self._prepare_task_params()
        self.generator.eval()
        with torch.no_grad():
            # 随机生成latent向量批次，并结合任务参数送入生成器
            z = torch.randn(self.config.trajectories_per_epoch, self.config.latent_dim).to(self.device)
            task_params_batch = task_params.repeat(self.config.trajectories_per_epoch, 1)
            trajectories_batch = self.generator(z, task_params_batch)
            for i in range(self.config.trajectories_per_epoch):
                trajectories.append(trajectories_batch[i].cpu().numpy())
        self.generator.train()
        return trajectories

    def _prepare_task_params(self):
        """准备任务相关参数（归一化的用户和目标位置）"""
        # 利用环境中已有的位置信息
        user_positions = self.env.user_positions.clone()
        target_positions = self.env.target_positions.clone()
        params = []
        # 添加用户位置（相对区域归一化）
        for user_pos in user_positions:
            params.extend((user_pos / self.config.area_size).tolist())
        # 添加目标位置（相对区域归一化）
        for target_pos in target_positions:
            params.extend((target_pos / self.config.area_size).tolist())
        return torch.tensor(params, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _optimize_resource_allocation(self, trajectory, traj_idx: int = 0):
        """针对给定轨迹，用SAC优化通信功率分配和感知决策"""
        rewards = []
        # —— ε-greedy 探索 ——
        epsilon = self.epsilon
        # 线性衰减
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)
        # 在单条轨迹上模拟多个独立episode交互
        for episode in range(self.config.episodes_per_trajectory):
            state = self.env.reset()
            episode_reward = 0
            for t in range(self.config.time_slots):
                # Epsilon-greedy exploration
                if np.random.random() < epsilon:
                    # 随机动作
                    action = self._random_action()
                else:
                    # 策略动作
                    action = self._select_action(state)
                next_state, reward, done, info = self.env.step(action, trajectory)
                # 将经验存入回放缓冲区
                self.reward_normalizer.update(reward)
                r_norm = self.reward_normalizer.normalize(reward)
                if isinstance(r_norm, torch.Tensor):
                    r_norm = r_norm.detach().cpu().item()
                r_norm = max(min(r_norm, 5.0), -5.0)
                self.replay_buffer.push(state, self._flatten_action(action), r_norm, next_state, done)
                # 经验累积到一定程度后开始更新SAC网络
                if len(self.replay_buffer) > self.config.batch_size:
                    for _ in range(self.gradient_steps):
                        self._update_sac()
                state = next_state
                episode_reward += reward
                if done:
                    break
            rewards.append(episode_reward)
            # 记录每个episode的奖励
            self.training_stats['rewards'].append(episode_reward)
            self._log_episode_row(
                epoch=self.current_epoch,
                traj_idx=traj_idx,
                episode_idx=episode,
                episode_reward=episode_reward,
                info=info  # 用最后一个 time step 的 info 作为该 episode 的收尾状态
            )

        return rewards

    def _random_action(self):
        """生成随机动作：含总发射功率"""
        # (A) 随机总功率（dBm 均匀采样 -> W）
        dbm = np.random.uniform(self.config.tx_power_min_dbm, self.config.tx_power_max_dbm)
        tx_power_w = float(10 ** ((dbm - 30.0) / 10.0))

        # (B) 分配：Dirichlet 产生(1+P)并归一（限制感知≤0.3）
        vec = np.random.rand(1 + self.config.num_users).astype(np.float32)
        power_alpha = vec / vec.sum()
        power_alpha[0] = min(power_alpha[0], 0.3)
        power_alpha = power_alpha / power_alpha.sum()
        power_alpha = torch.tensor(power_alpha, device=self.device, dtype=torch.float32)

        # (C) 感知：保留你原来的随机策略
        sensing = np.zeros(self.config.num_targets, dtype=np.float32)
        if np.random.random() > 0.3:
            sensing[np.random.randint(self.config.num_targets)] = 1.0
        sensing = torch.tensor(sensing, device=self.device, dtype=torch.float32)

        return {'tx_power_w': tx_power_w, 'power': power_alpha, 'sensing': sensing}

    def _select_action(self, state):
        """选择动作：第0维=总发射功率，其后(1+P)=分配系数，余下为感知离散logits"""
        state_tensor = state.clone().detach().unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            action, _, _ = self.policy.sample(state_tensor)
            action = action.squeeze(0)

        # === (A) 总发射功率：动态下限 + 爬坡限速 ===
        # 判定是否靠近且未完成目标，并计算SCNR债务比例
        uav_xy = self.env.uav_position[:2]
        tgt_xy = self.env.target_positions
        dist_all = torch.norm(tgt_xy - uav_xy.unsqueeze(0), dim=1)
        unfinished = (self.env.sensing_status == 0)
        scnr_thr_lin = 10 ** (self.config.scnr_threshold / 10)
        debt_ratio = 0.0
        near_unfinished = False
        if unfinished.any():
            idxs = unfinished.nonzero(as_tuple=False).squeeze(1)
            dists = dist_all[idxs]
            k = torch.argmin(dists)
            nearest_idx = idxs[k]
            near_unfinished = (dist_all[nearest_idx] <= self.config.sensing_range * 0.9)
            debt = max(0.0, (scnr_thr_lin - self.env.cumulative_scnr[nearest_idx]).item())
            debt_ratio = debt / scnr_thr_lin
        Pmin_comm = self.config.tx_power_min_dbm_comm
        Pmin_sense = self.config.tx_power_min_dbm_sense
        Pmax = self.config.tx_power_max_dbm
        a0 = action[0]
        base_min = Pmin_sense if near_unfinished else Pmin_comm
        # 债务越大 → 下限略抬高，最多+5 dB
        min_dbm = min(Pmax, base_min + 5.0 * debt_ratio)
        total_dbm = min_dbm + (a0 + 1.0) * 0.5 * (Pmax - min_dbm)
        # 爬坡限速
        if self.prev_total_dbm is not None:
            slew = self.config.tx_power_slew_db
            low = self.prev_total_dbm - slew
            high = self.prev_total_dbm + slew
            total_dbm = float(min(max(float(total_dbm), low), high))
            # 二次夹紧，确保不越界
        total_dbm = float(min(Pmax, max(min_dbm, total_dbm)))
        self.prev_total_dbm = float(total_dbm)
        tx_power_w = float(10 ** ((total_dbm - 30.0) / 10.0))  # dBm -> W

        # === (B) 连续分配：先决定 COMM/SENSE 的总份额（alpha），再在用户间归一化 ===
        # —— compute deficits (linear domain) ——
        scnr_thr_lin = 10 ** (self.config.scnr_threshold / 10)
        # average SINR so far (kept for features but no thresholding)
        denom = max(1, self.env.current_slot)
        avg_sinr_now = self.env.cumulative_sinr / denom
        sinr_def = 0.0  # SINR阈值约束移除
        scnr_def_vec = torch.clamp((scnr_thr_lin - self.env.cumulative_scnr) / max(scnr_thr_lin, 1e-6), min=0.0)
        # consider only unfinished targets for SCNR deficit
        mask_unfinished = (self.env.sensing_status == 0)
        if mask_unfinished.any():
            scnr_def = scnr_def_vec[mask_unfinished].mean().item()
        else:
            scnr_def = 0.0

        # logistic gating to get COMM share alpha in [alpha_min, alpha_max]
        a0 = float(self.config.alpha_a0);
        a1 = float(self.config.alpha_a1);
        a2 = float(self.config.alpha_a2)
        alpha_raw = a0 + a1 * float(sinr_def) - a2 * float(scnr_def)
        alpha_sig = float(torch.sigmoid(torch.tensor(alpha_raw)))
        alpha_min = float(self.config.alpha_min_frac);
        alpha_max = float(self.config.alpha_max_frac)
        alpha_comm = alpha_min + (alpha_max - alpha_min) * alpha_sig  # COMM total fraction

        # release SENSE power after SCNR satisfied for tau slots
        if int(getattr(self.env, 'scnr_done_streak', 0)) >= int(self.config.scnr_release_tau):
            sense_total_power = float(self.config.sense_power_floor_frac)
            comm_total_power = 1.0 - sense_total_power
        else:
            comm_total_power = float(alpha_comm)
            sense_total_power = 1.0 - comm_total_power

        # 用户内分配（加地板，防止饥饿）
        s = 1
        e = 1 + (1 + self.config.num_users)
        power_raw = action[s:e]
        power_exp = torch.exp(power_raw)
        power_norm = power_exp / (power_exp.sum(dim=0, keepdim=False) + 1e-8)

        if self.config.num_users > 0:
            eps = 1e-8
            power_floor = 1e-3
            comm_power_raw = F.relu(power_norm[1:self.config.num_users + 1]) + power_floor
            comm_power_sum = comm_power_raw.sum() + eps
            comm_powers = comm_power_raw / comm_power_sum * comm_total_power
        else:
            comm_powers = torch.tensor([], device=self.device)
        sensing_power = torch.tensor(sense_total_power, device=self.device)
        power = torch.cat([sensing_power.unsqueeze(0), comm_powers], dim=0)
        # === (C) 感知离散部分：余下 logits 仍按你的原逻辑筛选“在范围且未完成”的目标 ===
        logits = action[e:]
        sensing = torch.zeros(self.config.num_targets, device=self.device)
        if sensing_power > 0.01:
            uav_pos = self.env.uav_position[:2].clone().detach().to(self.device).float()
            target_pos = self.env.target_positions.clone().detach().to(self.device).float()
            dist = torch.norm(target_pos - uav_pos.unsqueeze(0), dim=1)
            status = self.env.sensing_status.clone().detach().to(self.device).long()
            valid_mask = (status == 0) & (dist <= self.config.sensing_range)
            valid_idxs = valid_mask.nonzero(as_tuple=False).squeeze(1)
            if valid_idxs.numel() > 0 and logits.numel() >= valid_idxs.numel():
                valid_logits = logits[:valid_idxs.numel()]
                best_idx_in_valid = torch.argmax(valid_logits)
                selected = valid_idxs[best_idx_in_valid]
                sensing[selected] = 1.0

        return {'tx_power_w': tx_power_w, 'power': power, 'sensing': sensing}

    def _flatten_action(self, action):
        power = action['power']
        sensing = action['sensing']
        txw = action.get('tx_power_w', 10 ** ((self.config.tx_power - 30.0) / 10.0))  # 兜底：用配置里的固定值
        if isinstance(power, torch.Tensor): power = power.detach().cpu().numpy()
        if isinstance(sensing, torch.Tensor): sensing = sensing.detach().cpu().numpy()
        return np.concatenate([[float(txw)], power.astype(np.float32), sensing.astype(np.float32)], axis=0)

    def _update_sac(self):
        """执行一次SAC网络参数更新"""
        state, action, reward, next_state, done = self.replay_buffer.sample(self.config.batch_size)
        # 转换为张量
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)
        # 更新Q网络：计算目标Q值并最小化TD误差
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state)
            q1_next = self.q1_target(next_state, next_action)
            q2_next = self.q2_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next) - self.log_alpha.exp() * next_log_prob
            target_q = reward + (1 - done) * self.config.gamma * q_next
            target_q = target_q.squeeze(-1)
            if target_q.dim() == 2 and target_q.size(1) > 1:
                target_q = target_q.min(dim=1, keepdim=True)[0]
        q1_loss = F.smooth_l1_loss(self.q1(state, action), target_q)
        q2_loss = F.smooth_l1_loss(self.q2(state, action), target_q)
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 0.5)
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 0.5)
        self.q2_optimizer.step()
        # 更新策略网络：最大化最小Q值和熵收益
        new_action, log_prob, _ = self.policy.sample(state)
        q1_new = self.q1(state, new_action)
        q2_new = self.q2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.log_alpha.exp() * log_prob - q_new).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.policy_optimizer.step()
        # 更新温度参数α
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_alpha.clamp_(np.log(self.config.alpha_min), 2)
        # 记录SAC损失
        self.training_stats['q_loss'].append((q1_loss.item() + q2_loss.item()) / 2)
        self.training_stats['policy_loss'].append(policy_loss.item())
        self.training_stats['alpha_loss'].append(alpha_loss.item())

    def _soft_update(self, source, target):
        """软更新目标Q网络参数"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

    def _traj_to_state_action(self, trajectories: torch.Tensor):
        B, T, D = trajectories.shape
        flat = trajectories.view(B, -1)  # [B, traj_flat_dim]
        action_batch = self.traj_encoder(flat)  # [B, critic_in]
        state_batch = torch.empty(B, self.config.state_dim,
                                  device=flat.device)  # 或者 zeros
        return state_batch, action_batch

    def _update_gan(self, trajectories, rewards):
        """更新GAN生成器和判别器"""
        # 在 update_gan 开头，计算当前的自适应 λ_rl
        warmup = self.config.rl_warmup_steps
        lam_max = self.config.lambda_rl
        t = self.current_epoch
        if t < warmup:
            lambda_rl = lam_max * (t / warmup)
        else:
            lambda_rl = lam_max

        rewards = np.array(rewards)
        threshold = np.percentile(rewards, 60)  # 选择高于60百分位的轨迹作为“好轨迹”
        good_indices = np.where(rewards >= threshold)[0]
        if len(good_indices) == 0:
            return  # 若无轨迹达到阈值则本轮不更新GAN
        # 将“好轨迹”转为张量
        good_trajectories = torch.tensor(np.array([trajectories[i] for i in good_indices]),
                                         dtype=torch.float32).to(self.device)
        # 训练判别器
        for _ in range(self.config.n_critic):
            d_loss_real = -self.discriminator(good_trajectories).mean()
            z = torch.randn(good_trajectories.size(0), self.config.latent_dim).to(self.device)
            task_params = self._prepare_task_params().repeat(good_trajectories.size(0), 1)
            fake_trajectories = self.generator(z, task_params)
            d_loss_fake = self.discriminator(fake_trajectories.detach()).mean()
            gp = self._gradient_penalty(good_trajectories, fake_trajectories)
            d_loss = d_loss_real + d_loss_fake + self.config.lambda_gp * gp
            self.d_optimizer.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
            self.d_optimizer.step()
        # 训练生成器
        z = torch.randn(self.config.batch_size, self.config.latent_dim).to(self.device)
        task_params = self._prepare_task_params().repeat(self.config.batch_size, 1)
        fake_trajectories = self.generator(z, task_params)
        g_loss_adv = -self.discriminator(fake_trajectories).mean()
        g_loss_phys = self._physics_loss(fake_trajectories)
        g_loss_task = self._task_loss(fake_trajectories)
        # —— 新增：RL 指导项 ——
        # 对一小批 generator 生成的轨迹，跑一次 SAC 资源分配，得出 avg_reward，
        # 然后最大化该 avg_reward（即 loss = -avg_reward）
        # RL 指导项（只在指定间隔或小批量上执行）
        # 减小采样次数（比如只采样 8~16 条轨迹）
        # 1. 冻结 Critic 参数，仅做前向
        for p in self.q1.parameters(): p.requires_grad = False
        for p in self.q2.parameters(): p.requires_grad = False
        # 2. 把 fake_trajectories 转为 Critic 要的 (state, action)
        state_batch, action_batch = self._traj_to_state_action(fake_trajectories)

        # 3. 前向 Q 网络，取双 Q 最小值
        q1_vals = self.q1(state_batch, action_batch)
        q2_vals = self.q2(state_batch, action_batch)
        q_vals = torch.min(q1_vals, q2_vals)

        # 4. 定义 RL‐loss：最大化 Q，所以取负号
        g_loss_rl = - q_vals.mean()
        # —— 合并 loss ——
        g_loss = g_loss_adv + self.config.lambda_phys * g_loss_phys + self.config.lambda_task * g_loss_task + lambda_rl * g_loss_rl
        self.g_optimizer.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 0.5)
        self.g_optimizer.step()
        for p in self.q1.parameters(): p.requires_grad = True
        for p in self.q2.parameters(): p.requires_grad = True
        # 记录GAN损失
        self.training_stats['g_loss'].append(g_loss.item())
        self.training_stats['g_loss_rl'].append(g_loss_rl.item())
        self.training_stats['d_loss'].append(d_loss.item())

    def _gradient_penalty(self, real_data, fake_data):
        """计算WGAN的梯度惩罚项"""
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1).to(self.device)
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)
        prob_interpolated = self.discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=prob_interpolated, inputs=interpolated,
            grad_outputs=torch.ones_like(prob_interpolated),
            create_graph=True, retain_graph=True
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()
        return gradient_penalty

    def _physics_loss(self, trajectories):
        """计算物理约束损失：速度/加速度/边界约束"""
        dt = self.config.slot_duration
        velocity = torch.diff(trajectories, dim=1) / dt
        v_norm = torch.norm(velocity, dim=2)
        v_min_loss = F.relu(self.config.v_min - v_norm).mean()
        v_max_loss = F.relu(v_norm - self.config.v_max).mean()
        acceleration = torch.diff(velocity, dim=1) / dt
        a_norm = torch.norm(acceleration, dim=2)
        a_max_loss = F.relu(a_norm - self.config.a_max).mean()
        boundary_loss = F.relu(-trajectories).mean() + F.relu(trajectories - self.config.area_size).mean()
        return v_min_loss + v_max_loss + a_max_loss + boundary_loss

    def _task_loss(self, trajectories):
        """计算任务相关损失：目标未覆盖惩罚和起终点约束"""
        loss = 0.0

        time_steps = trajectories.shape[1]
        target_positions = self.env.target_positions.float().to(self.device)
        user_positions = self.env.user_positions.float().to(self.device)
        # 目标覆盖损失：距离每个目标最小距离超过感知范围的惩罚，鼓励轨迹靠近目标
        target_coverage_loss = 0.0
        for q in range(self.config.num_targets):
            target_pos = target_positions[q]
            distances = torch.norm(trajectories - target_pos.unsqueeze(0).unsqueeze(0), dim=2)
            min_distances, _ = torch.min(distances, dim=1)
            target_coverage_loss = F.relu(min_distances - self.config.sensing_range).mean()
        loss += 10.0 * target_coverage_loss

        # 2. 通信质量损失 - 在非感知期间优化用户SINR
        comm_quality_loss = 0.0
        for p in range(self.config.num_users):
            user_pos = user_positions[p]
            distances = torch.norm(trajectories - user_pos.unsqueeze(0).unsqueeze(0), dim=2)

            # 计算每个时间步的通信质量（距离越近越好）
            # 使用反比例函数，距离越近奖励越大
            comm_quality = 1.0 / (distances / 100.0 + 1.0)  # 归一化距离
            comm_quality_loss += torch.mean(comm_quality)

        # 通信质量损失（取负值，因为我们要最大化通信质量）
        loss -= 0.5 * comm_quality_loss / self.config.num_users

        # 3. 能耗优化损失
        energy_loss = 0.0

        # 3.1 速度变化损失（鼓励平滑运动）
        velocity = (trajectories[:, 1:] - trajectories[:, :-1]) / self.config.slot_duration
        velocity_magnitude = torch.norm(velocity, dim=2)

        # 惩罚过高的速度（能耗与速度平方成正比）
        speed_penalty = torch.mean(velocity_magnitude ** 2)
        energy_loss += 0.3 * speed_penalty

        # 3.2 加速度损失（鼓励平滑加速）
        acceleration = (velocity[:, 1:] - velocity[:, :-1]) / self.config.slot_duration
        acceleration_magnitude = torch.norm(acceleration, dim=2)

        # 惩罚急剧的加速度变化
        accel_penalty = torch.mean(acceleration_magnitude ** 2)
        energy_loss += 0.4 * accel_penalty

        # 3.3 方向变化损失（鼓励直线运动）
        if time_steps > 2:
            direction_changes = []
            for t in range(1, time_steps - 1):
                dir1 = trajectories[:, t] - trajectories[:, t - 1]
                dir2 = trajectories[:, t + 1] - trajectories[:, t]

                # 归一化方向向量
                dir1_norm = dir1 / (torch.norm(dir1, dim=1, keepdim=True) + 1e-6)
                dir2_norm = dir2 / (torch.norm(dir2, dim=1, keepdim=True) + 1e-6)

                # 计算方向变化（余弦相似度，越接近1越好）
                direction_similarity = torch.sum(dir1_norm * dir2_norm, dim=1)
                direction_changes.append(1.0 - direction_similarity)

            if direction_changes:
                direction_change_loss = torch.mean(torch.stack(direction_changes))
                energy_loss += 0.2 * direction_change_loss

        loss += energy_loss

        # 计算在目标附近的时间比例
        target_vicinity_time = 0
        for q in range(self.config.num_targets):
            target_pos = target_positions[q]
            distances = torch.norm(trajectories - target_pos.unsqueeze(0).unsqueeze(0), dim=2)
            in_range = (distances <= self.config.sensing_range).float()
            target_vicinity_time += torch.sum(in_range, dim=1)

        # 计算在用户附近的时间比例
        user_vicinity_time = 0
        for p in range(self.config.num_users):
            user_pos = user_positions[p]
            distances = torch.norm(trajectories - user_pos.unsqueeze(0).unsqueeze(0), dim=2)
            near_user = (distances <= 300.0).float()  # 300米内认为是良好通信范围
            user_vicinity_time += torch.sum(near_user, dim=1)

        # 平衡感知和通信时间
        total_time = time_steps
        sensing_ratio = target_vicinity_time / total_time
        comm_ratio = user_vicinity_time / total_time

        # 鼓励合理的时间分配（不要过度偏向某一方）
        balance_penalty = torch.abs(sensing_ratio - 0.4) + torch.abs(comm_ratio - 0.6)
        balance_loss = torch.mean(balance_penalty)
        loss += 0.3 * balance_loss

        # 起点和终点约束损失：轨迹首末点距起点的距离（希望起点和终点重合）
        # 6. 基地返回损失（软约束）
        if time_steps > 10:
            # 最后10%的时间应该朝向基地移动
            end_portion = max(1, time_steps // 10)
            end_trajectory = trajectories[:, -end_portion:]
            base_pos = torch.tensor([600.0, 1000.0], device=trajectories.device)

            # 计算最后阶段到基地的距离变化
            distances_to_base = torch.norm(end_trajectory - base_pos, dim=2)
            # 应该随时间递减
            for t in range(1, end_portion):
                distance_reduction = distances_to_base[:, t - 1] - distances_to_base[:, t]
                # 鼓励朝基地移动
                return_loss = F.relu(-distance_reduction).mean()
                loss += 0.1 * return_loss
        return loss / self.config.num_targets

    def _log_training_stats(self, epoch):
        """打印训练过程中的统计信息"""
        print(f"\n--- Epoch {epoch}/{self.config.max_epochs} ---")
        if len(self.training_stats['g_loss']) > 0:
            print(f"Generator Loss: {np.mean(self.training_stats['g_loss'][-10:]):.4f}")
            print(f"Discriminator Loss: {np.mean(self.training_stats['d_loss'][-10:]):.4f}")
        if len(self.training_stats['q_loss']) > 0:
            print(f"Q Loss: {np.mean(self.training_stats['q_loss'][-100:]):.4f}")
            print(f"Policy Loss: {np.mean(self.training_stats['policy_loss'][-100:]):.4f}")
            print(f"Alpha: {self.log_alpha.exp().item():.4f}")
        if len(self.training_stats['rewards']) > 0:
            rewards = torch.stack(self.training_stats['rewards'][-100:])
            avg_reward = rewards.mean().item()
            print(f"Average Reward: {avg_reward:.4f}")
        if len(self.training_stats['sinr_satisfaction']) > 0:
            print(f"SCNR Satisfaction: {self.training_stats['scnr_satisfaction'][-1] * 100:.1f}%")
            print(f"SCNR λ: {getattr(self, 'lambda_scnr', 0.0):.3f}")
        if len(self.training_stats['energy_consumption']) > 0:
            print(f"Average Energy: {np.mean(self.training_stats['energy_consumption'][-100:]):.2f} J")
        if len(self.training_stats['coverage_rate']) > 0:
            print(f"Coverage Rate: {self.training_stats['coverage_rate'][-1] * 100:.1f}%")
            # 新增：输出当前BER估计
            if hasattr(self.env, 'cumulative_sinr') and self.env.current_slot > 0:
                avg_sinr = self.env.cumulative_sinr / self.env.current_slot
                print(
                    f"Current Average SINR per User: {[f'{sinr:.2f} dB' for sinr in (10 * torch.log10(avg_sinr)).tolist()]}")

                # 计算并显示当前BER估计
                current_ber_per_user = []
                for p in range(self.config.num_users):
                    ber = self.env._ber_pred(avg_sinr[p]).item()
                    current_ber_per_user.append(ber)

                print(f"Estimated BER per User: {[f'{ber:.2e}' for ber in current_ber_per_user]}")
                print(f"System Average BER: {np.mean(current_ber_per_user):.2e}")

    def evaluate(self, num_episodes=10):
        """评估智能体在随机轨迹下的平均性能"""
        total_rewards = []
        sinr_satisfaction = []
        scnr_satisfaction = []
        per_user_sinr_satisfaction = []
        all_episode_bers = []  # 新增：存储所有episode的BER
        for episode in range(num_episodes):
            # 对每个episode随机生成一个轨迹并执行
            z = torch.randn(1, self.config.latent_dim).to(self.device)
            task_params = self._prepare_task_params()
            self.generator.eval()
            with torch.no_grad():
                trajectory = self.generator(z, task_params).cpu().numpy()[0]
            state = self.env.reset()
            episode_reward = 0
            # 记录每个时间步的SINR
            all_sinr = []
            for t in range(self.config.time_slots):
                with torch.no_grad():
                    action = self._select_action(state)
                next_state, reward, done, info = self.env.step(action, trajectory)
                episode_reward += reward
                all_sinr.append(info['sinr'])
                state = next_state
                if done:
                    break

            # 计算每个用户的平均SINR
            if len(all_sinr) > 0:
                sinr_tensor = torch.stack(all_sinr, dim=0)
                all_sinr = sinr_tensor.detach().cpu().numpy()
                avg_sinr_per_user = np.mean(all_sinr, axis=0)  # 每个用户的平均SINR

                # 计算每个用户的BER
                episode_bers = []
                for p in range(self.config.num_users):
                    sinr_linear = avg_sinr_per_user[p]
                    ber = self.env._ber_pred(torch.tensor(sinr_linear)).item()
                    episode_bers.append(ber)
                all_episode_bers.append(episode_bers)
                print(f"Episode {episode + 1} BER per user: {[f'{ber:.2e}' for ber in episode_bers]}")
            else:
                avg_sinr_per_user = np.zeros(self.config.num_users)
            # 检查每个用户的平均SINR约束
            user_sinr_satisfied = [0.0 for _ in range(self.config.num_users)]  # SINR阈值相关已移除
            per_user_sinr_satisfaction.append(user_sinr_satisfied)
            sinr_satisfaction.append(0.0)

            # 检查SCNR约束
            scnr_threshold_linear = 10 ** (self.config.scnr_threshold / 10)
            scnr_sat = sum(info['cumulative_scnr'] >= scnr_threshold_linear) / self.config.num_targets
            scnr_satisfaction.append(scnr_sat)
            total_rewards.append(episode_reward)
            print(f"[Eval] Episode {episode + 1}: slots used = {int(info.get('slots_used', self.config.time_slots))}")

            rewards_cpu = [r.detach().cpu().item() if isinstance(r, torch.Tensor) else r
                           for r in total_rewards]
            sinr_sat_cpu = [x.detach().cpu().item() if isinstance(x, torch.Tensor) else x
                            for x in sinr_satisfaction]
            scnr_sat_cpu = [x.detach().cpu().item() if isinstance(x, torch.Tensor) else x
                            for x in scnr_satisfaction]
            per_user_array = np.stack(per_user_sinr_satisfaction, axis=0)
        results = {
            'average_reward': np.mean(rewards_cpu),
            'sinr_satisfaction_rate': np.mean(sinr_sat_cpu),
            'per_user_sinr_satisfaction': np.mean(per_user_array, axis=0),
            'scnr_satisfaction_rate': np.mean(scnr_sat_cpu)
        }
        # 计算并输出平均BER
        if all_episode_bers:
            avg_ber_per_user = np.mean(all_episode_bers, axis=0)
            print(f"\n=== BER Results ===")
            for p in range(self.config.num_users):
                print(f"User {p + 1} Average BER: {avg_ber_per_user[p]:.2e}")
            print(f"Overall Average BER: {np.mean(avg_ber_per_user):.2e}")

            # 添加到返回结果中
            results['average_ber_per_user'] = avg_ber_per_user
            results['overall_average_ber'] = np.mean(avg_ber_per_user)
        return results

    def save_models(self, path):
        """保存模型"""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'log_alpha': self.log_alpha,
            'training_stats': self.training_stats
        }, path)

    def load_models(self, path):
        """加载模型"""
        checkpoint = torch.load(path)

        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])
        self.log_alpha = checkpoint['log_alpha']
        self.training_stats = checkpoint['training_stats']


# 经验回放缓冲区类
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # 确保所有数据都是tensor
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device, dtype=torch.float32)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32)
        if isinstance(reward, torch.Tensor):
            reward = reward.item()

        # 存储在GPU上
        self.buffer.append((
            state.clone(),
            action.clone(),
            reward,
            next_state.clone(),
            done
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        # 直接stack tensor（已经在GPU上）
        states = torch.stack([item[0] for item in batch])
        actions = torch.stack([item[1] for item in batch])
        rewards = torch.tensor([item[2] for item in batch], device=self.device, dtype=torch.float32).unsqueeze(1)
        next_states = torch.stack([item[3] for item in batch])
        dones = torch.tensor([item[4] for item in batch], device=self.device, dtype=torch.float32).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# 改进的轨迹生成器（生成无人机轨迹）
class TrajectoryGenerator(nn.Module):
    """改进的轨迹生成器，利用全连接+LSTM+注意力机制生成轨迹"""

    def __init__(self, config: SystemConfig):
        super(TrajectoryGenerator, self).__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.sensing_status = [0 for _ in range(self.config.num_targets)]
        # 输入维度：潜在向量 + 任务参数
        input_dim = config.latent_dim + 2 * (config.num_users + config.num_targets)
        # 网络层定义
        self.fc1 = nn.Linear(input_dim, 768)
        self.fc2 = nn.Linear(768, 1536)
        self.fc3 = nn.Linear(1536, 1024)
        self.lstm = nn.LSTM(1024, 512, num_layers=2, batch_first=True)
        self.attention = nn.MultiheadAttention(512, num_heads=8)
        self.fc4 = nn.Linear(512, 256)
        self.output_layer = nn.Linear(256, 2)
        self.bn1 = nn.BatchNorm1d(768)
        self.bn2 = nn.BatchNorm1d(1536)
        self.bn3 = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(0.1)

    def forward(self, z, task_params):
        """前向生成轨迹"""
        batch_size = z.size(0)
        # 将随机潜在向量与任务参数拼接
        x = torch.cat([z, task_params], dim=1)
        # 三层全连接+激活
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        # 扩展维度进入LSTM：重复输入作为每个时间步的特征向量
        x = x.unsqueeze(1).repeat(1, self.config.time_slots, 1)
        lstm_out, _ = self.lstm(x)
        # 自注意力机制，捕获轨迹序列长程依赖
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        # 输出两层全连接，得到轨迹坐标序列
        x = F.relu(self.fc4(attn_out))
        trajectory = self.output_layer(x)
        # 将输出经Sigmoid映射到区域范围内（0-area_size）
        trajectory = torch.sigmoid(trajectory) * self.config.area_size
        trajectory = self._apply_constraints_and_guidance(trajectory, task_params)
        return trajectory

    def _calculate_required_sensing_time(self, distance, target_idx):
        """
        根据SCNR阈值和距离动态计算所需的感知时间
        Args:
            distance: UAV到目标的距离 (m)
            target_idx: 目标索引
        Returns:
            required_steps: 所需的时间步数
            estimated_scnr_per_step: 估算的每步SCNR增量
        """
        device = next(self.parameters()).device
        scnr_threshold_linear = torch.tensor(10 ** (self.config.scnr_threshold / 10), device=device)
        tx_dbm = torch.tensor(getattr(self.config, 'tx_power_min_dbm', self.config.tx_power), device=device)
        tx_power_linear = 10 ** ((tx_dbm - 30) / 10)
        noise_power_linear = torch.tensor(10 ** ((self.config.noise_power - 30) / 10), device=device)
        # 路径损耗和增益估算
        wavelength = torch.tensor(3e8 / self.config.carrier_freq, device=device)
        antenna_gain = torch.sqrt(torch.tensor(self.config.antenna_num, device=device))
        sigma_rcs = torch.tensor(1.0, device=device)
        # 计算路径损耗系数（基于双程路径损耗）
        beta_q = torch.sqrt(wavelength ** 2 * sigma_rcs / ((4 * torch.pi) ** 3 * distance ** 4))

        # 估算每时间步的SCNR增量
        # 考虑30%的感知功率分配（典型的通信感知功率分配策略）
        sense_share_est = torch.tensor(0.45, device=device)
        base_scnr_per_step = sense_share_est * tx_power_linear * antenna_gain ** 2 * beta_q ** 2 / noise_power_linear

        # 考虑实际系统中的各种损耗因素:
        # 1. 杂波干扰: 降低50%
        # 2. 信道估计误差: 降低20%
        # 3. 系统实现损失: 降低15%
        realistic_factor = torch.tensor(0.5 * 0.8 * 0.85, device=device)  # 总体降低约66%
        estimated_scnr_per_step = base_scnr_per_step * realistic_factor

        # 计算达到阈值所需的时间步数
        required_steps = torch.ceil(scnr_threshold_linear / estimated_scnr_per_step.clamp(min=1e-10))
        # print(f"Required Steps: {required_steps}")

        # 考虑实际情况，添加50%的安全裕度以应对:
        # 1. 功率分配的动态变化
        # 2. 信道条件的时变性
        # 3. 多目标感知的干扰
        # 4. UAV机动对天线指向的影响
        required_steps = int(required_steps * 2.0)

        return required_steps, estimated_scnr_per_step

    def _apply_constraints_and_guidance(self, trajectory, task_params):
        """应用物理约束并引导轨迹按顺序访问目标"""
        dt = self.config.slot_duration
        batch_size, time_steps, _ = trajectory.shape

        # 解析任务参数中的用户和目标位置
        num_users = self.config.num_users
        num_targets = self.config.num_targets

        # 用户位置
        user_positions = []
        for i in range(num_users):
            user_x = task_params[:, 2 * i] * self.config.area_size
            user_y = task_params[:, 2 * i + 1] * self.config.area_size
            user_positions.append(torch.stack([user_x, user_y], dim=1))

        # 目标位置
        target_positions = []
        for i in range(num_targets):
            target_x = task_params[:, 2 * num_users + 2 * i] * self.config.area_size
            target_y = task_params[:, 2 * num_users + 2 * i + 1] * self.config.area_size
            target_positions.append(torch.stack([target_x, target_y], dim=1))

        # 构建新的轨迹
        trajectory_new = torch.zeros_like(trajectory)
        # 起始位置：基地坐标
        start_pos = torch.tensor([600.0, 1000.0], device=trajectory.device).expand(batch_size, -1)
        trajectory_new[:, 0] = start_pos

        # 当前位置和时间
        current_pos = start_pos.clone()
        current_time = 0

        # 访问每个目标
        for target_idx in range(num_targets):
            if self.sensing_status[target_idx] == 1:
                continue
            if current_time >= time_steps - 10:  # 预留时间返回基地
                break

            target_pos = target_positions[target_idx]

            # 阶段1：接近目标
            while current_time < time_steps - 10:
                current_time += 1
                if current_time >= time_steps:
                    break

                # 计算到目标的方向和距离
                direction = target_pos - current_pos
                distance = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-6)
                direction_normalized = direction / distance

                # 检查附近用户（距离小于500米）
                near_users = []
                for user_pos in user_positions:
                    user_dist = torch.norm(current_pos - user_pos, dim=1)
                    if torch.mean(user_dist) < 300:  # 300米范围内
                        near_users.append(user_pos)

                # 如果附近有用户且距离目标较远（>300米），调整方向靠近用户
                if near_users and torch.mean(distance) > 200:
                    # 计算平均用户方向
                    avg_user_dir = torch.zeros_like(direction_normalized)
                    for user_pos in near_users:
                        user_dir = user_pos - current_pos
                        user_dir = user_dir / torch.norm(user_dir, dim=1, keepdim=True)
                        avg_user_dir += user_dir
                    avg_user_dir = avg_user_dir / len(near_users)
                    avg_user_dir = avg_user_dir / torch.norm(avg_user_dir, dim=1, keepdim=True)

                    # 混合方向：70%朝向目标，30%朝向用户
                    mixed_dir = 0.7 * direction_normalized + 0.3 * avg_user_dir
                    mixed_dir = mixed_dir / torch.norm(mixed_dir, dim=1, keepdim=True)
                    direction_normalized = mixed_dir

                # 根据距离调整速度
                # 距离远时高速，接近目标时减速
                speed = torch.where(
                    distance > self.config.sensing_range * 1.5,
                    self.config.v_max * torch.ones_like(distance),
                    torch.where(
                        distance > self.config.sensing_range * 0.8,
                        self.config.v_min + (self.config.v_max - self.config.v_min) *
                        (distance - self.config.sensing_range * 0.8) / (self.config.sensing_range * 0.7),
                        self.config.v_min * torch.ones_like(distance)
                    )
                )

                # 计算速度向量
                velocity = direction_normalized * speed

                # 应用速度约束
                v_norm = torch.norm(velocity, dim=1, keepdim=True)
                v_norm_clamped = torch.clamp(v_norm, self.config.v_min, self.config.v_max)

                mask = v_norm > 1e-6
                velocity = torch.where(
                    mask,
                    velocity / v_norm * v_norm_clamped,
                    direction_normalized * self.config.v_min
                )
                # 更新位置
                current_pos = current_pos + velocity * dt
                trajectory_new[:, current_time] = current_pos

                # 检查是否进入感知范围
                dist_to_target = torch.norm(current_pos - target_pos, dim=1)
                if torch.mean(dist_to_target) < self.config.sensing_range * 0.9:
                    # 阶段2：根据SCNR阈值动态计算所需的感知时间
                    avg_distance = torch.mean(dist_to_target).item()
                    required_steps, estimated_scnr_per_step = self._calculate_required_sensing_time(avg_distance,
                                                                                                    target_idx)

                    # 限制在合理范围内，确保不超过剩余时间
                    max_available_time = time_steps - current_time - 10
                    sensing_time = min(max(required_steps, 8), max_available_time,
                                       30)  # 最少8步，最多30步(一个全局硬限制，绝对不会分配超过 30 个时隙给感知。)

                    # 生成基础盘旋路径（椭圆形状）
                    num_hover_points = sensing_time * 2  # 增加点密度使轨迹更平滑

                    # 创建角度张量
                    angles = torch.linspace(0, 2 * torch.pi, num_hover_points, device=target_pos.device)

                    # 使用广播机制一次性计算所有点的偏移量
                    radius_x = self.config.sensing_range * 0.4
                    radius_y = self.config.sensing_range * 0.3

                    # 添加随机扰动使路径自然
                    rand_offset = torch.randn(batch_size, num_hover_points, 2,
                                              device=target_pos.device) * self.config.sensing_range * 0.05

                    # 计算椭圆点 - 使用广播机制
                    offset_x = radius_x * torch.cos(angles).view(1, -1, 1)
                    offset_y = radius_y * torch.sin(angles).view(1, -1, 1)
                    offset = torch.cat([offset_x, offset_y], dim=2)

                    # 扩展目标位置以匹配偏移量形状
                    target_pos_expanded = target_pos.unsqueeze(1).expand(-1, num_hover_points, -1)

                    # 计算所有点位置
                    hover_points = target_pos_expanded + offset + rand_offset

                    # 应用样条插值使路径更平滑
                    hover_points_np = hover_points.cpu().numpy()
                    t = np.linspace(0, 1, num_hover_points)

                    # 为每个批次创建平滑路径
                    smooth_path_list = []
                    for b in range(batch_size):
                        # 提取当前批次的盘旋点
                        points_b = hover_points_np[b]

                        # 创建样条曲线
                        cs_x = CubicSpline(t, points_b[:, 0])
                        cs_y = CubicSpline(t, points_b[:, 1])

                        # 生成平滑路径
                        num_smooth_points = sensing_time
                        t_smooth = np.linspace(0, 1, num_smooth_points)
                        smooth_x = cs_x(t_smooth)
                        smooth_y = cs_y(t_smooth)

                        smooth_path_list.append(np.stack([smooth_x, smooth_y], axis=1))

                    smooth_path_np = np.array(smooth_path_list)
                    smooth_path = torch.tensor(smooth_path_np, device=trajectory.device)

                    # 应用Savitzky-Golay滤波器进一步平滑
                    if num_smooth_points >= 3:  # 只有数据点足够多时才进行滤波
                        window_size = min(7, num_smooth_points)
                        # 确保窗口大小为奇数且不超过数据点数量
                        window_size = min(window_size, num_smooth_points)
                        if window_size % 2 == 0:  # 如果是偶数，减1使其为奇数
                            window_size = max(3, window_size - 1)  # 确保至少为3

                        if window_size >= 3 and window_size <= num_smooth_points:
                            smooth_path_np_smoothed = smooth_path_np.copy()
                            for i in range(batch_size):
                                try:
                                    smooth_path_np_smoothed[i, :, 0] = savgol_filter(
                                        smooth_path_np[i, :, 0], window_size, 3, mode='interp')
                                    smooth_path_np_smoothed[i, :, 1] = savgol_filter(
                                        smooth_path_np[i, :, 1], window_size, 3, mode='interp')
                                except ValueError:
                                    # 如果滤波失败，使用原始数据
                                    smooth_path_np_smoothed[i, :, 0] = smooth_path_np[i, :, 0]
                                    smooth_path_np_smoothed[i, :, 1] = smooth_path_np[i, :, 1]
                            smooth_path = torch.tensor(smooth_path_np_smoothed, device=trajectory.device)
                        else:
                            smooth_path = torch.tensor(smooth_path_np, device=trajectory.device)
                    else:
                        # 数据点太少，直接使用原始路径
                        smooth_path = torch.tensor(smooth_path_np, device=trajectory.device)

                    # 确保所有点在感知范围内
                    for i in range(num_smooth_points):
                        dist_to_target = torch.norm(smooth_path[:, i] - target_pos, dim=1)
                        mask_out_of_range = dist_to_target > self.config.sensing_range * 0.9
                        if torch.any(mask_out_of_range):
                            # 缩放超出范围的点
                            scale = (self.config.sensing_range * 0.9) / dist_to_target
                            offset = smooth_path[:, i] - target_pos
                            smooth_path[:, i] = target_pos + offset * scale.unsqueeze(1)

                    # 跟随平滑路径
                    for point_idx in range(num_smooth_points):
                        current_time += 1
                        if current_time >= time_steps:
                            break

                        # 应用速度约束
                        if point_idx > 0:
                            displacement = smooth_path[:, point_idx] - smooth_path[:, point_idx - 1]
                            required_velocity = displacement / dt
                            v_norm = torch.norm(required_velocity, dim=1, keepdim=True)
                            max_hover_speed = self.config.v_min * 1.5
                            if torch.any(v_norm > max_hover_speed):
                                scale = max_hover_speed / v_norm
                                smooth_path[:, point_idx] = smooth_path[:, point_idx - 1] + displacement * scale

                        current_pos = smooth_path[:, point_idx]
                        trajectory_new[:, current_time] = current_pos
                    break

        # 返回基地
        remaining_time = time_steps - current_time - 1
        if remaining_time > 0:
            for t in range(remaining_time):
                current_time += 1
                if current_time >= time_steps:
                    break

                progress = t / max(remaining_time, 1)
                t_smooth = progress * progress * (3 - 2 * progress)
                trajectory_new[:, current_time] = (1 - t_smooth) * current_pos + t_smooth * start_pos

        return trajectory_new


# 轨迹判别器：判别真实轨迹和生成轨迹
class TrajectoryDiscriminator(nn.Module):
    """轨迹判别器"""

    def __init__(self, config: SystemConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.time_slots * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(128)

    def forward(self, trajectory):
        """判别轨迹真实或生成"""
        x = trajectory.view(trajectory.size(0), -1)
        x = F.leaky_relu(self.ln1(self.fc1(x)), 0.2)
        x = F.leaky_relu(self.ln2(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.ln3(self.fc3(x)), 0.2)
        validity = self.fc4(x)
        return validity


# 高斯策略网络（输出连续动作：功率分配和感知决策）
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        hidden_dim = 256
        self.fc1 = nn.Linear(self.state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, self.action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, self.action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(min=-20, max=2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        """采样动作（重参数化）"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


# Q网络（双Q结构）
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


# 可视化工具类：提供轨迹绘制和训练曲线绘制方法
class Visualizer:
    def plot_trajectory(self, trajectory, user_positions=None, target_positions=None, save_path='trajectory.png'):
        if isinstance(trajectory, torch.Tensor):
            trajectory = trajectory.cpu().numpy()
        if isinstance(user_positions, torch.Tensor):
            user_positions = user_positions.cpu().numpy()
        if isinstance(target_positions, torch.Tensor):
            target_positions = target_positions.cpu().numpy()
        plt.figure(figsize=(8, 8))
        plt.style.use('default')
        plt.rcParams['font.size'] = 12
        user_positions = np.array(user_positions, dtype=float)
        target_positions = np.array(target_positions, dtype=float)
        if target_positions.ndim != 2 or target_positions.shape[1] < 2:
            raise ValueError(f"plot_trajectory: target_positions 必须是形状 (N,2) 数组，当前是 {target_positions.shape}")

        # 设置坐标轴和网格
        ax = plt.gca()

        trajectory_line = plt.plot(trajectory[:, 0], trajectory[:, 1],
                                   color='#2E86AB', linewidth=2, alpha=0.8,
                                   label='UAV Trajectory', zorder=3)

        # 添加轨迹方向箭头
        if len(trajectory) > 10:
            for i in range(0, len(trajectory) - 1, len(trajectory) // 8):
                dx = trajectory[i + 1, 0] - trajectory[i, 0]
                dy = trajectory[i + 1, 1] - trajectory[i, 1]
                plt.arrow(trajectory[i, 0], trajectory[i, 1], dx * 0.3, dy * 0.3,
                          head_width=15, head_length=20, fc='#2E86AB', ec='#2E86AB', alpha=0.6)

        # 绘制起点和终点 - 更醒目的标记
        plt.plot(trajectory[0, 0], trajectory[0, 1], 'o', color='#A23B72',
                 markersize=18, markeredgecolor='white', markeredgewidth=2,
                 label='Start Point', zorder=5)
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], 's', color='#F18F01',
                 markersize=12, markeredgecolor='white', markeredgewidth=2,
                 label='End Point', zorder=5)

        # 绘制用户位置 - 使用不同颜色和更大的标记
        user_colors = ['#4ECDC4', '#45B7D1', '#96CEB4']  # 青色系渐变
        for i, pos in enumerate(user_positions):
            color = user_colors[i % len(user_colors)]
            plt.scatter(pos[0], pos[1], c=color, marker='^', s=200,
                        edgecolors='white', linewidth=2, label=f'User U{i + 1}' if i == 0 else "",
                        zorder=4)

            # 添加用户标签
            plt.annotate(f'U{i + 1}', (pos[0], pos[1]),
                         xytext=(8, 8), textcoords='offset points',
                         fontsize=11, fontweight='bold', color='white',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
                         zorder=6)

        # 绘制目标位置 - 使用星形标记和渐变色
        target_colors = ['#E74C3C', '#C0392B', '#A93226', '#922B21', '#7B241C']  # 红色系渐变
        for i, pos in enumerate(target_positions):
            color = target_colors[i % len(target_colors)]
            plt.scatter(pos[0], pos[1], c=color, marker='*', s=200,
                        edgecolors='white', linewidth=2, label=f'Target T{i + 1}' if i == 0 else "",
                        zorder=4)

            # 添加目标标签
            plt.annotate(f'T{i + 1}', (pos[0], pos[1]),
                         xytext=(10, 10), textcoords='offset points',
                         fontsize=11, fontweight='bold', color='white',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
                         zorder=6)

            # 为每个目标添加感知范围圆圈 - 美化样式
            sensing_circle = Circle((pos[0], pos[1]), 80,
                                    fill=False, color=color, linestyle='--',
                                    linewidth=2, alpha=0.6, zorder=2)
            plt.gca().add_patch(sensing_circle)

            # 添加填充的感知范围（半透明）
            sensing_fill = Circle((pos[0], pos[1]), 80,
                                  fill=True, color=color, alpha=0.1, zorder=1)
            plt.gca().add_patch(sensing_fill)

        # 添加基地标记
        base_pos = [600, 1000]  # 基地位置
        plt.scatter(base_pos[0], base_pos[1], c='#8E44AD', marker='H', s=250,
                    edgecolors='white', linewidth=2, label='Base Station', zorder=4)
        plt.annotate('Start/End Point', base_pos,
                     xytext=(0, 15), textcoords='offset points',
                     fontsize=12, fontweight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#8E44AD', alpha=0.8),
                     ha='center', zorder=6)

        # 设置坐标轴范围为 0-1200
        plt.xlim(0, 1200)
        plt.ylim(0, 1200)
        plt.xlabel('x (m)', fontsize=14)
        plt.ylabel('y (m)', fontsize=14)

        # 设置网格
        ax.grid(True, alpha=0.3)
        # 设置坐标轴刻度标签字体
        ax.tick_params(axis='both', which='major', labelsize=12)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('normal')

        # 创建自定义图例
        legend_elements = [
            Line2D([0], [0], color='#2E86AB', lw=3, label='UAV Trajectory'),
            Line2D([0], [0], marker='H', color='w', markerfacecolor='#A23B72',
                   markersize=10, label='Start Point'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#F18F01',
                   markersize=10, label='End Point'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#4ECDC4',
                   markersize=12, label='Users'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='#E74C3C',
                   markersize=15, label='Targets'),
            Line2D([0], [0], color='#E74C3C', lw=2, linestyle='--',
                   alpha=0.6, label='Sensing Projection Region')
        ]

        # 将图例放在图内右下角
        plt.legend(handles=legend_elements, loc='lower right',
                   fontsize=12, framealpha=0.9, fancybox=False, shadow=False,
                   frameon=True, facecolor='white', edgecolor='black')

        # 调整布局
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()  # 确保关闭图形

    @staticmethod
    def plot_training_curves(training_stats, save_path=None):
        rewards = training_stats.get('rewards', [])
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.detach().cpu().numpy()
        elif isinstance(rewards, list):
            rewards = np.array([x.detach().cpu().item() if isinstance(x, torch.Tensor) else x for x in rewards])

        if len(rewards) == 0:
            print("没有奖励数据可绘制。")
            return

        plt.figure(figsize=(8, 6))

        # 平滑奖励曲线
        window_size = max(1, min(100, len(rewards) // 10))
        if window_size > 1:
            smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
            plt.plot(smoothed_rewards, label='Smoothed Rewards', linewidth=2, color='purple')

        # 原始奖励曲线
        plt.plot(rewards, alpha=0.3, label='Raw Rewards', color='gray')

        plt.title('Episode Rewards', fontsize=14)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        # plt.close()


# 主函数：训练GAN-SAC模型并评估性能
def main():
    set_seed(42)
    config = SystemConfig()
    gan_sac = GANSAC(config)
    print("Using device:", torch.device(config.device),
          " – cuda.is_available():", torch.cuda.is_available())
    print("Starting GAN-SAC training...")
    gan_sac.train()
    gan_sac.export_episodes_csv('episodes.csv')

    print("\nEvaluating performance...")
    results = gan_sac.evaluate(num_episodes=20)
    print("\n=== Evaluation Results ===")
    print(f"Average Reward: {results['average_reward']:.4f}")
    print(f"SCNR Satisfaction Rate: {results['scnr_satisfaction_rate'] * 100:.2f}%")
    # 新增：输出BER结果
    if 'average_ber_per_user' in results:
        print(f"\n=== Bit Error Rate Results ===")
        for p in range(config.num_users):
            print(f"User {p + 1} Average BER: {results['average_ber_per_user'][p]:.2e}")
        print(f"Overall System BER: {results['overall_average_ber']:.2e}")

        # 输出BER性能评级
        overall_ber = results['overall_average_ber']
        if overall_ber < 1e-5:
            ber_grade = "Excellent"
        elif overall_ber < 1e-4:
            ber_grade = "Good"
        elif overall_ber < 1e-3:
            ber_grade = "Fair"
        else:
            ber_grade = "Poor"
        print(f"BER Performance Grade: {ber_grade}")

    gan_sac.save_models('optimized_gan_sac_model.pth')
    # 生成并保存轨迹图
    visualizer = Visualizer()
    gan_sac.generator.eval()
    z = torch.randn(1, config.latent_dim).to(gan_sac.device)
    task_params = gan_sac._prepare_task_params()
    with torch.no_grad():
        trajectory = gan_sac.generator(z, task_params).cpu().numpy()[0]
    visualizer.plot_trajectory(
        trajectory=trajectory,
        user_positions=gan_sac.env.user_positions,
        target_positions=gan_sac.env.target_positions,
        save_path='trajectory.png'
    )
    # 绘制训练曲线
    visualizer.plot_training_curves(gan_sac.training_stats, 'training_curves.png')
    # 新增：OTFS vs OFDM BER比较
    print(f"\n=== OTFS vs OFDM BER Comparison ===")
    try:
        otfs_ber, ofdm_ber = compare_otfs_ofdm_ber(
            gan_sac.env.otfs,
            scheme=config.ber_scheme,
            snrs=range(0, 15, 3),
            frames=100
        )

        print("OTFS BER Results:")
        for snr, ber in otfs_ber.items():
            print(f"  SNR {snr} dB: BER = {ber:.2e}")

        print("OFDM BER Results:")
        for snr, ber in ofdm_ber.items():
            print(f"  SNR {snr} dB: BER = {ber:.2e}")

    except Exception as e:
        print(f"BER comparison failed: {e}")


if __name__ == "__main__":
    main()


# === Convolutional Code (rate 1/2, K=7, 171/133) ===
def _parity_int(x: int) -> int:
    return (bin(x).count('1') & 1)


class ConvCodeR12K7:
    G0 = 0o171
    G1 = 0o133
    K = 7
    MASK = (1 << K) - 1

    @staticmethod
    def encode(info_bits):
        import torch
        state = 0
        outs = []
        for b in info_bits.tolist():
            state = ((state << 1) | int(b)) & ConvCodeR12K7.MASK
            o0 = _parity_int(state & ConvCodeR12K7.G0)
            o1 = _parity_int(state & ConvCodeR12K7.G1)
            outs.extend([o0, o1])
        for _ in range(ConvCodeR12K7.K - 1):
            state = ((state << 1) & ConvCodeR12K7.MASK)
            o0 = _parity_int(state & ConvCodeR12K7.G0)
            o1 = _parity_int(state & ConvCodeR12K7.G1)
            outs.extend([o0, o1])
        return torch.tensor(outs, dtype=torch.int8, device=info_bits.device)

    @staticmethod
    def viterbi_decode(llrs_pair):
        import torch
        K = ConvCodeR12K7.K
        S = 1 << (K - 1)
        T = llrs_pair.numel() // 2
        next_state = [[0, 0] for _ in range(S)]
        out_bits = [[(0, 0), (0, 0)] for _ in range(S)]
        for s in range(S):
            for b in (0, 1):
                st = ((s << 1) | b) & ((1 << K) - 1)
                o0 = _parity_int(st & ConvCodeR12K7.G0)
                o1 = _parity_int(st & ConvCodeR12K7.G1)
                next_state[s][b] = st & (S - 1)
                out_bits[s][b] = (o0, o1)
        INF = 1e9
        pm = torch.full((T + 1, S), INF, device=llrs_pair.device)
        bp = torch.full((T + 1, S), -1, dtype=torch.int16, device=llrs_pair.device)
        pm[0, 0] = 0.0
        for t in range(T):
            L0 = float(llrs_pair[2 * t])
            L1 = float(llrs_pair[2 * t + 1])
            for s in range(S):
                val = float(pm[t, s].item())
                if val >= INF / 2:
                    continue
                for b in (0, 1):
                    ns = next_state[s][b]
                    o0, o1 = out_bits[s][b]
                    metric = -((1 - 2 * o0) * L0 + (1 - 2 * o1) * L1) / 2.0
                    cand = val + metric
                    if cand < pm[t + 1, ns]:
                        pm[t + 1, ns] = cand
                        bp[t + 1, ns] = (s << 1) | b
        s_best = 0
        info_rev = []
        for t in range(T, 0, -1):
            val = int(bp[t, s_best].item())
            s_prev = val >> 1
            b = val & 1
            info_rev.append(b)
            s_best = s_prev
        info_bits = torch.tensor(info_rev[::-1], dtype=torch.int8, device=llrs_pair.device)
        return info_bits[:-(K - 1)] if info_bits.numel() > (K - 1) else torch.zeros(0, dtype=torch.int8,
                                                                                    device=llrs_pair.device)


def compare_otfs_ofdm_ber(otfs_processor, scheme='QPSK', snrs=range(-2, 13, 2), frames=200, code='conv_r12_k7'):
    otfs_res = otfs_processor.evaluate_ber_coded(scheme, snrs, frames=frames, code=code)
    ofdm_res = otfs_processor.evaluate_ber_coded_ofdm(scheme, snrs, frames=frames, code=code)
    print("OTFS :", otfs_res)
    print("OFDM :", ofdm_res)
    return otfs_res, ofdm_res

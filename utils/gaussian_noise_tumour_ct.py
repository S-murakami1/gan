import math
import torch
import numpy as np
from monai.config import KeysCollection
from monai.transforms.compose import MapTransform
from torch import clone as clone


class GaussianNoiseTumourCT(MapTransform):
    """
    CT画像のラベルからバウンディングボックスを動的に計算し、
    腫瘍領域にガウシアンノイズを付加するトランスフォーム。

    BraTSの GaussianNoiseTumour との主な相違点:
        - CSVの事前計算済みバウンディングボックス情報が不要 (ラベルから動的算出)
        - 出力キー名が scan_ct_* に統一
        - 強度正規化はクロップ後のミンマックス [-1, 1] (元コードと同方式)
          入力CTが [0, 1] 正規化済みであることを前提とし、HUウィンドウ処理は行わない

    入力キー: scan_ct (CT画像, shape: [C, H, W, D], 値域: [0, 1])
              label    (腫瘍マスク, shape: [C, H, W, D], 値: 0 or 1)

    出力キー (d に追加):
        scan_ct_crop      : 腫瘍周囲をクロップした正規化済みCT
        scan_ct_crop_pad  : patch_size^3にパディングしたクロップCT [-1, 1]
        scan_ct_noisy     : 腫瘍領域にガウシアンノイズを付加したもの [-1, 1]
        label_crop_pad    : patch_size^3にパディングしたクロップラベル
        label_crop        : クロップラベル（パディングなし）

    Args:
        keys (KeysCollection): CT画像のキー名（例: "scan_ct"）
        patch_size (int): クロップ後のパッチサイズ (default: 96)
    """

    def __init__(
        self,
        keys: KeysCollection,
        patch_size: int = 96,
    ):
        super().__init__(keys)
        self.keys = keys
        self.patch_size = patch_size

    def __call__(self, data):
        d = dict(data)
        scan_ct = d[self.keys]
        label = d["label"]
        _, max_x, max_y, max_z = scan_ct.shape

        # ── ラベルから腫瘍バウンディングボックスを算出 ────────────────────────
        label_np = np.array(label)
        fg = np.any(label_np > 0, axis=0)  # (H, W, D)
        xs, ys, zs = np.where(fg)

        if len(xs) == 0:
            # 腫瘍が存在しない場合: 画像中心を使用
            cx, cy, cz = max_x // 2, max_y // 2, max_z // 2
            x_min, x_max = cx, cx
            y_min, y_max = cy, cy
            z_min, z_max = cz, cz
        else:
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            z_min, z_max = int(zs.min()), int(zs.max())

        x_size = x_max - x_min
        y_size = y_max - y_min
        z_size = z_max - z_min

        # ── 腫瘍周囲を patch_size にパディング込みでクロップ ─────────────────
        P = self.patch_size

        x_extreme_dif = x_size
        y_extreme_dif = y_size
        z_extreme_dif = z_size

        x_pad = (P - x_extreme_dif) / 2
        y_pad = (P - y_extreme_dif) / 2
        z_pad = (P - z_extreme_dif) / 2

        C_x = -0.5 if x_pad < 0 else 0.5
        C_y = -0.5 if y_pad < 0 else 0.5
        C_z = -0.5 if z_pad < 0 else 0.5

        x_base = x_min - int(x_pad)
        x_top  = x_max + int(x_pad + C_x)
        y_base = y_min - int(y_pad)
        y_top  = y_max + int(y_pad + C_y)
        z_base = z_min - int(z_pad)
        z_top  = z_max + int(z_pad + C_z)

        # ── 境界外に出た分のパディング量を計算 ─────────────────────────────
        x_base_pad = y_base_pad = z_base_pad = 0
        x_top_pad  = y_top_pad  = z_top_pad  = 0

        if x_base < 0:
            x_base_pad = -x_base
            x_base = 0
        if y_base < 0:
            y_base_pad = -y_base
            y_base = 0
        if z_base < 0:
            z_base_pad = -z_base
            z_base = 0
        if x_top > max_x:
            x_top_pad = x_top - max_x
            x_top = max_x
        if y_top > max_y:
            y_top_pad = y_top - max_y
            y_top = max_y
        if z_top > max_z:
            z_top_pad = z_top - max_z
            z_top = max_z

        # ── クロップ ──────────────────────────────────────────────────────
        # 元コードと同様: クロップ後にミンマックス [-1, 1] 正規化
        scan_ct_crop = scan_ct[:, x_base:x_top, y_base:y_top, z_base:z_top]
        scan_ct_crop = self._rescale_array_tensor(scan_ct_crop, minv=-1, maxv=1)
        label_crop   = label_np[:, x_base:x_top, y_base:y_top, z_base:z_top]

        # ── パディング (境界値: scan=-1, label=0) ───────────────────────────
        pw = ((0, 0), (x_base_pad, x_top_pad), (y_base_pad, y_top_pad), (z_base_pad, z_top_pad))
        scan_ct_crop_pad = np.pad(scan_ct_crop, pad_width=pw, mode="constant", constant_values=(-1, -1))
        label_crop_pad   = np.pad(label_crop,   pad_width=pw, mode="constant", constant_values=(0,  0))

        # ── 腫瘍領域へのガウシアンノイズ付加 ────────────────────────────────
        max_size = max(x_size, y_size, z_size)
        exp_base = self._norm_exp_base(max_size)
        scan_ct_noisy = self._add_gaussian_noise_tumour(
            scan=scan_ct_crop_pad, label=label_crop_pad, exp_base=exp_base
        )
        scan_ct_noisy = self._rescale_array_numpy(scan_ct_noisy, minv=-1, maxv=1)

        # ── データ辞書に格納 ──────────────────────────────────────────────
        d[self.keys]          = scan_ct  # 元スキャンをそのまま保持
        d["scan_ct_crop"]     = scan_ct_crop
        d["scan_ct_crop_pad"] = scan_ct_crop_pad
        d["scan_ct_noisy"]    = scan_ct_noisy
        d["label_crop"]       = label_crop
        d["label_crop_pad"]   = label_crop_pad

        return d

    # ── ヘルパー ──────────────────────────────────────────────────────────

    def _rescale_array_tensor(self, arr, minv: float, maxv: float):
        """元コードの rescale_array と同等 (Tensor / numpy 両対応)"""
        mina = torch.min(arr) if isinstance(arr, torch.Tensor) else np.min(arr)
        maxa = torch.max(arr) if isinstance(arr, torch.Tensor) else np.max(arr)
        if mina == maxa:
            return arr * minv
        norm = (arr - mina) / (maxa - mina)
        return (norm * (maxv - minv)) + minv

    def _rescale_array_numpy(self, arr: np.ndarray, minv: float, maxv: float) -> np.ndarray:
        mina, maxa = np.min(arr), np.max(arr)
        if mina == maxa:
            return arr * minv
        norm = (arr - mina) / (maxa - mina)
        return (norm * (maxv - minv)) + minv

    def _norm_exp_base(self, value: float) -> float:
        """腫瘍サイズ (28〜96) をガウシアンノイズの指数ベース (1.1〜1.3) にマッピング"""
        m = -0.2 / 68
        c = 1.1 - 96 * m
        return m * value + c

    def _add_gaussian_noise_tumour(
        self, scan: np.ndarray, label: np.ndarray, exp_base: float
    ) -> np.ndarray:
        """腫瘍マスク内のボクセルにガウシアンノイズを付加する"""
        P = self.patch_size
        scan_noisy = np.copy(scan)
        noise = np.full((1, P, P, P), 1000.0, dtype=np.float32)

        for x in range(P):
            for y in range(P):
                for z in range(P):
                    if np.any(label[:, x, y, z] > 0):
                        noise[0, x, y, z] = float(torch.randn(1))

        np.copyto(
            scan_noisy,
            noise,
            where=np.logical_and(noise < 100, scan_noisy != -1),
        )
        return scan_noisy

import os
import warnings
import torch
import numpy as np
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.data.utils import pad_list_data_collate
from monai.transforms import (
    Compose,
    DeleteItemsd,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ToTensord,
)
from utils.gaussian_noise_tumour_ct import GaussianNoiseTumourCT


def get_ct_loader(args):
    """
    nifti/default_spacing の CT データセット用データローダーを返す。

    期待するディレクトリ構造:
        <data_root>/
            train_val.json       # Decathlonフォーマット
            images/
                <id>_data.nii.gz
            labels/
                <id>_label.nii.gz

    train_val.json の "training" リストを学習データとして使用する。
    バリデーションが必要な場合は "validation" キーを参照。

    args に期待するフィールド:
        args.data_root   (str)  : データセットのルートディレクトリ
        args.json_path   (str)  : train_val.jsonのパス (空文字のとき data_root 直下を自動検索)
        args.batch_size  (int)  : バッチサイズ (Generator + Discriminator 分として 2倍で DataLoader に渡す)
        args.num_workers (int)  : DataLoader のワーカー数
        args.noise_type  (str)  : "gaussian_tumour" のみ対応 (将来拡張用)
        args.patch_size  (int)  : クロップパッチサイズ (default:  96)
        args.cache_rate  (float): CacheDataset のキャッシュ割合 (default: 1.0)
        args.split       (str)  : "training" or "validation" (default: "training")

    Returns:
        DataLoader: (batch_size*2) で iterate される DataLoader
            各バッチに含まれるキー（学習に必要な3キーのみキャッシュ）:
                "scan_ct_crop_pad" : 腫瘍周囲96^3クロップ+パディング [-1, 1]
                "scan_ct_noisy"    : ノイズ付加済み96^3パッチ [-1, 1]
                "label_crop_pad"   : 腫瘍周囲96^3クロップ+パディング (0/1)
            ※ 元スキャン全体 (scan_ct, label) はキャッシュ前に削除する。
              これにより推定メモリ使用量: 247 GB → 約 5.4 GB (545症例)
    """
    NUM_WORKERS = int(args.num_workers)
    SPLIT = getattr(args, "split", "training")
    PATCH_SIZE = int(getattr(args, "patch_size", 96))
    CACHE_RATE = float(getattr(args, "cache_rate", 1.0))

    # ── JSON パスの解決 ───────────────────────────────────────────────────
    data_root = args.data_root
    json_path = getattr(args, "json_path", "")
    if json_path == "":
        json_path = os.path.join(data_root, "train_val.json")
    print(f"JSON_PATH  : {json_path}")
    print(f"DATA_ROOT  : {data_root}")
    print(f"SPLIT      : {SPLIT}")
    print(f"PATCH_SIZE : {PATCH_SIZE}")

    # ── ファイルリストの読み込み ──────────────────────────────────────────
    data_list = load_decathlon_datalist(
        data_list_file_path=json_path,
        is_segmentation=True,
        data_list_key=SPLIT,
        base_dir=data_root,
    )
    print(f"Number of {SPLIT} samples: {len(data_list)}")

    # load_decathlon_datalist は {"image": ..., "label": ...} を返すが、
    # 本コードベースのキー規約に合わせて "scan_ct" / "label" にリネームする
    data_list = [{"scan_ct": d["image"], "label": d["label"]} for d in data_list]

    # ── トランスフォーム定義 ─────────────────────────────────────────────
    train_transforms = Compose(
        [
            LoadImaged(keys=["scan_ct", "label"], image_only=False),
            EnsureChannelFirstd(keys=["scan_ct", "label"]),
            EnsureTyped(keys=["scan_ct", "label"]),
            GaussianNoiseTumourCT(
                keys="scan_ct",
                patch_size=PATCH_SIZE,
            ),
            # 学習に不要な元スキャン全体（232MB/症例）をキャッシュ前に削除し
            # メモリ使用量を 247GB → 約5GB に削減する
            DeleteItemsd(keys=["scan_ct", "label", "scan_ct_crop", "label_crop"]),
            ToTensord(
                keys=[
                    "scan_ct_crop_pad",
                    "scan_ct_noisy",
                    "label_crop_pad",
                ]
            ),
        ]
    )

    # ── データセット ─────────────────────────────────────────────────────
    estimated_gb = (96**3 * 4 * 3 * len(data_list)) / 1024**3
    print(f"キャッシュ推定メモリ: {estimated_gb * CACHE_RATE:.1f} GB (cache_rate={CACHE_RATE})")
    if estimated_gb * CACHE_RATE > 32:
        warnings.warn(
            f"推定キャッシュサイズ {estimated_gb * CACHE_RATE:.1f} GB が大きいです。"
            " --cache_rate を下げることを検討してください。"
        )
    train_ds = CacheDataset(
        data=data_list,
        transform=train_transforms,
        cache_rate=CACHE_RATE,
        copy_cache=False,
        progress=True,
        num_workers=NUM_WORKERS,
    )

    # Generator と Discriminator に別々のサンプルを渡すため batch_size*2 で取得
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size * 2),
        num_workers=NUM_WORKERS,
        drop_last=True,
        shuffle=True,
        collate_fn=pad_list_data_collate,
    )
    print(f"Dataset {SPLIT}: number of batches: {len(train_loader)}")
    print("CT データローダーの準備完了。")
    return train_loader

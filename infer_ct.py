import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset, load_decathlon_datalist
from monai.networks.nets import AttentionUnet, SwinUNETR, UNet
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, ToTensord

from utils.gaussian_noise_tumour_ct import GaussianNoiseTumourCT


def build_generator(args):
    if args.generator_type == "SwinUNETR":
        model = SwinUNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            use_checkpoint=args.use_checkpoint,
        )
    elif args.generator_type == "AttentionUnet":
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=(48, 96, 192, 384, 768),
            strides=(2, 2, 2, 2, 1),
            kernel_size=3,
            up_kernel_size=3,
            dropout=0.0,
        )
    elif args.generator_type == "Unet":
        model = UNet(
            spatial_dims=3,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=(48, 96, 192, 384, 768),
            strides=(2, 2, 2, 1),
        )
    else:
        raise ValueError("generator_type must be one of: SwinUNETR, AttentionUnet, Unet")
    return model


def save_nifti(image_tensor, out_path):
    image_np = np.squeeze(image_tensor.detach().cpu().numpy())
    nib.save(nib.Nifti1Image(image_np, affine=np.eye(4)), out_path)


def extract_case_name(path_str):
    name = Path(path_str).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(path_str).stem


def main():
    parser = argparse.ArgumentParser(description="CT generator inference for a few cases")
    parser.add_argument("--weights_path", required=True, type=str, help="Path to generator_*.pt")
    parser.add_argument("--data_root", required=True, type=str, help="CT dataset root (contains train_val.json)")
    parser.add_argument("--json_path", default="", type=str, help="Path to train_val.json (empty=auto)")
    parser.add_argument("--split", default="training", type=str, help="Decathlon split key")
    parser.add_argument("--num_cases", default=5, type=int, help="Number of cases to run")
    parser.add_argument("--batch_size", default=1, type=int, help="Inference batch size")
    parser.add_argument("--num_workers", default=0, type=int, help="Dataloader workers")
    parser.add_argument("--patch_size", default=96, type=int, help="Patch size for CT preprocessing")
    parser.add_argument(
        "--no_ct_rescale_patch",
        dest="ct_rescale_patch",
        action="store_false",
        default=True,
        help="Match train.py: disable patch min-max (default: ON).",
    )
    parser.add_argument("--out_dir", default="inference_ct", type=str, help="Output directory")

    # Model args kept aligned with train.py defaults.
    parser.add_argument("--generator_type", default="SwinUNETR", type=str)
    parser.add_argument("--in_channels", default=2, type=int, help="CT uses scan_noisy(1)+label(1)=2")
    parser.add_argument("--out_channels", default=1, type=int)
    parser.add_argument("--feature_size", default=48, type=int)
    parser.add_argument("--use_checkpoint", action="store_true")
    args = parser.parse_args()

    if args.json_path == "":
        args.json_path = os.path.join(args.data_root, "train_val.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_list = load_decathlon_datalist(
        data_list_file_path=args.json_path,
        is_segmentation=True,
        data_list_key=args.split,
        base_dir=args.data_root,
    )
    data_list = [
        {
            "scan_ct": d["image"],
            "label": d["label"],
            "case_name": extract_case_name(d["image"]),
        }
        for d in data_list
    ][: args.num_cases]
    if len(data_list) == 0:
        raise RuntimeError("No cases found. Check --json_path and --split.")

    transforms = Compose(
        [
            LoadImaged(keys=["scan_ct", "label"], image_only=False),
            EnsureChannelFirstd(keys=["scan_ct", "label"]),
            EnsureTyped(keys=["scan_ct", "label"]),
            GaussianNoiseTumourCT(
                keys="scan_ct",
                patch_size=args.patch_size,
                rescale_patch=args.ct_rescale_patch,
            ),
            ToTensord(keys=["scan_ct_crop_pad", "scan_ct_noisy", "label_crop_pad"]),
        ]
    )
    loader = DataLoader(
        Dataset(data=data_list, transform=transforms),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    generator = build_generator(args).to(device)
    ckpt = torch.load(args.weights_path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    generator.load_state_dict(state_dict, strict=True)
    generator.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        sample_idx = 0
        for batch in loader:
            x_crop_pad = batch["scan_ct_crop_pad"].to(device)
            x_crop_noisy = batch["scan_ct_noisy"].to(device)
            y_crop_pad = batch["label_crop_pad"].to(device)
            case_names = batch["case_name"]
            input_noise = torch.cat([x_crop_noisy, y_crop_pad], dim=1)
            scan_recon = torch.clamp(generator(input_noise), 0.0, 1.0)

            bs = scan_recon.shape[0]
            for i in range(bs):
                if sample_idx >= args.num_cases:
                    break
                case_name = str(case_names[i])
                save_nifti(x_crop_pad[i], str(out_dir / f"{case_name}_scan_ct_crop_pad.nii.gz"))
                save_nifti(y_crop_pad[i], str(out_dir / f"{case_name}_label_crop_pad.nii.gz"))
                save_nifti(x_crop_noisy[i], str(out_dir / f"{case_name}_scan_ct_noisy.nii.gz"))
                save_nifti(scan_recon[i], str(out_dir / f"{case_name}_scan_recon.nii.gz"))
                sample_idx += 1
            if sample_idx >= args.num_cases:
                break

    print(f"Saved {sample_idx} cases to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()


import argparse
import os
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from LabelGAN import (
    Code_Discriminator as LabelCodeDiscriminator,
    Discriminator as LabelDiscriminator,
    Encoder as LabelEncoder,
    Generator as LabelGenerator,
)
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.data.utils import pad_list_data_collate
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, Resized, ToTensord
from monai.transforms.compose import MapTransform


class CropLabelCT(MapTransform):
    """Crop tumour bbox from CT label and pad to patch_size^3."""

    def __init__(self, keys, patch_size=96):
        super().__init__(keys)
        self.patch_size = patch_size

    def __call__(self, data):
        d = dict(data)
        label = np.asarray(d["label"])
        _, max_x, max_y, max_z = label.shape
        fg = np.any(label > 0, axis=0)
        xs, ys, zs = np.where(fg)

        if len(xs) == 0:
            cx, cy, cz = max_x // 2, max_y // 2, max_z // 2
            x_min = x_max = cx
            y_min = y_max = cy
            z_min = z_max = cz
        else:
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            z_min, z_max = int(zs.min()), int(zs.max())

        x_size, y_size, z_size = x_max - x_min, y_max - y_min, z_max - z_min
        p = self.patch_size
        x_pad = (p - x_size) / 2
        y_pad = (p - y_size) / 2
        z_pad = (p - z_size) / 2
        c_x = -0.5 if x_pad < 0 else 0.5
        c_y = -0.5 if y_pad < 0 else 0.5
        c_z = -0.5 if z_pad < 0 else 0.5

        x_base = x_min - int(x_pad)
        x_top = x_max + int(x_pad + c_x)
        y_base = y_min - int(y_pad)
        y_top = y_max + int(y_pad + c_y)
        z_base = z_min - int(z_pad)
        z_top = z_max + int(z_pad + c_z)

        x_base_pad = y_base_pad = z_base_pad = 0
        x_top_pad = y_top_pad = z_top_pad = 0
        if x_base < 0:
            x_base_pad, x_base = -x_base, 0
        if y_base < 0:
            y_base_pad, y_base = -y_base, 0
        if z_base < 0:
            z_base_pad, z_base = -z_base, 0
        if x_top > max_x:
            x_top_pad, x_top = x_top - max_x, max_x
        if y_top > max_y:
            y_top_pad, y_top = y_top - max_y, max_y
        if z_top > max_z:
            z_top_pad, z_top = z_top - max_z, max_z

        crop = label[:, x_base:x_top, y_base:y_top, z_base:z_top]
        crop_pad = np.pad(
            crop,
            ((0, 0), (x_base_pad, x_top_pad), (y_base_pad, y_top_pad), (z_base_pad, z_top_pad)),
            mode="constant",
            constant_values=0,
        )
        d["label_crop_pad"] = crop_pad
        return d


def save_sample(image, out_path):
    arr = np.squeeze(image.detach().cpu().numpy())
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine=np.eye(4)), str(out_path))


def create_train_loader(data_root, json_path, split, batch_size, num_workers, patch_size, cache_rate):
    if json_path == "":
        json_path = os.path.join(data_root, "train_val.json")
    data_list = load_decathlon_datalist(json_path, True, split, data_root)
    data_list = [{"label": x["label"]} for x in data_list]

    transforms = Compose(
        [
            LoadImaged(keys=["label"], image_only=False),
            EnsureChannelFirstd(keys=["label"]),
            EnsureTyped(keys=["label"]),
            CropLabelCT(keys=["label"], patch_size=patch_size),
            Resized(keys=["label_crop_pad"], spatial_size=(64, 64, 64), mode="nearest"),
            ToTensord(keys=["label_crop_pad"], dtype=torch.float32),
        ]
    )
    ds = CacheDataset(data_list, transforms, cache_rate=cache_rate, copy_cache=False, progress=True, num_workers=num_workers)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, collate_fn=pad_list_data_collate)


def main():
    parser = argparse.ArgumentParser(description="CT label GAN training")
    parser.add_argument("--logdir", default="label_ct", type=str)
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--json_path", default="", type=str)
    parser.add_argument("--split", default="training", type=str)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--patch_size", default=96, type=int)
    parser.add_argument("--cache_rate", default=1.0, type=float)
    parser.add_argument("--total_iter", default=200000, type=int)
    parser.add_argument("--resume_iter", default=0, type=int)
    parser.add_argument("--latent_dim", default=100, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    args = parser.parse_args()

    home = Path("Checkpoint") / args.logdir / "label_ct"
    (home / "weights").mkdir(parents=True, exist_ok=True)
    (home / "loss_lists").mkdir(parents=True, exist_ok=True)
    (home / "checkpoint_scans").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = create_train_loader(args.data_root, args.json_path, args.split, args.batch_size, args.num_workers, args.patch_size, args.cache_rate)
    loader_iter = iter(loader)

    g = LabelGenerator(noise=args.latent_dim, out_channels=1).to(device)
    e = LabelEncoder(out_class=args.latent_dim, in_channels=1).to(device)
    d = LabelDiscriminator(in_channels=1).to(device)
    cd = LabelCodeDiscriminator(code_size=args.latent_dim, num_units=4096).to(device)
    g_opt = optim.AdamW(g.parameters(), lr=args.lr)
    e_opt = optim.AdamW(e.parameters(), lr=args.lr)
    d_opt = optim.AdamW(d.parameters(), lr=args.lr)
    cd_opt = optim.AdamW(cd.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    losses = {"gen_enc": [], "disc": [], "code_disc": [], "mse": []}

    for step in range(args.resume_iter, args.total_iter + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        real = batch["label_crop_pad"].to(device)
        bsz = real.size(0)
        z_rand = torch.randn((bsz, args.latent_dim), device=device)

        # train E/G
        for p in d.parameters():
            p.requires_grad = False
        for p in cd.parameters():
            p.requires_grad = False
        e_opt.zero_grad()
        g_opt.zero_grad()
        z_hat = e(real).view(bsz, -1)
        x_hat = g(z_hat)
        x_rand = g(z_rand)
        cd_z_hat = cd(z_hat).mean()
        d_mix = d(x_hat).mean() + d(x_rand).mean()
        mse_loss = mse(x_hat, real)
        ge_loss = -cd_z_hat - d_mix + 100.0 * mse_loss
        ge_loss.backward()
        e_opt.step()
        g_opt.step()

        # train D
        for p in d.parameters():
            p.requires_grad = True
        d_opt.zero_grad()
        d_loss = -2.0 * d(real).mean() + d(x_hat.detach()).mean() + d(x_rand.detach()).mean()
        d_loss.backward()
        d_opt.step()

        # train CD
        for p in cd.parameters():
            p.requires_grad = True
        cd_opt.zero_grad()
        cd_loss = -cd(z_rand.detach()).mean() + cd(z_hat.detach()).mean()
        cd_loss.backward()
        cd_opt.step()

        losses["gen_enc"].append(float(ge_loss.detach().cpu()))
        losses["disc"].append(float(d_loss.detach().cpu()))
        losses["code_disc"].append(float(cd_loss.detach().cpu()))
        losses["mse"].append(float(mse_loss.detach().cpu()))
        if step % 100 == 0:
            print(f"[{step}/{args.total_iter}] ge={losses['gen_enc'][-1]:.4f} d={losses['disc'][-1]:.4f} cd={losses['code_disc'][-1]:.4f} mse={losses['mse'][-1]:.4f}")

        if step % 1000 == 0 and step > 0:
            torch.save({"iteration": step, "state_dict": g.state_dict(), "optimizer": g_opt.state_dict()}, home / "weights" / f"G_iter_{step}.pt")
            torch.save({"iteration": step, "state_dict": d.state_dict(), "optimizer": d_opt.state_dict()}, home / "weights" / f"D_iter_{step}.pt")
            torch.save({"iteration": step, "state_dict": e.state_dict(), "optimizer": e_opt.state_dict()}, home / "weights" / f"E_iter_{step}.pt")
            torch.save({"iteration": step, "state_dict": cd.state_dict(), "optimizer": cd_opt.state_dict()}, home / "weights" / f"CD_iter_{step}.pt")
            save_sample(real[0], home / "checkpoint_scans" / f"{step}_real.nii.gz")
            save_sample(x_rand[0], home / "checkpoint_scans" / f"{step}_x_rand.nii.gz")
            with open(home / "loss_lists" / "losses.pkl", "wb") as fp:
                pickle.dump(losses, fp)

    print("Finished CT label GAN training.")


if __name__ == "__main__":
    main()


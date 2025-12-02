import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm

import op_utils
from data import CustomDataLoader, MVTecDataset, SimpleDataset
from data.feature_provider import WideFeatures


class VariationalEncoder(nn.Module):
    def __init__(self, latent=32, nf=128, kernel_size=1):
        super(VariationalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(512, nf, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(nf, nf, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        self.to_mu = nn.Conv2d(nf, latent, kernel_size=1)
        self.to_std = nn.Conv2d(nf, latent, kernel_size=1)

        self.kl = 0

    def forward(self, features):
        out = self.encoder(features)
        mu = self.to_mu(out)
        sigma = torch.exp(self.to_std(out))

        z = mu + sigma * torch.randn(mu.shape, device=features.device)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).mean()

        return z

    def inference(self, features):
        out = self.encoder(features)
        mu = self.to_mu(out)
        return mu


class Decoder(nn.Module):
    def __init__(self, latent=32, nf=128, kernel_size=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(latent, nf, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(nf, nf, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(nf, 512, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class VAE(nn.Module):
    def __init__(self, latent=128, nf=512, kernel_size=1):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(latent, nf, kernel_size=kernel_size)
        self.decoder = Decoder(latent, nf, kernel_size=kernel_size)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

    def inference(self, x):
        return self.decoder(self.encoder.inference(x))


def train_vae(dataset, path, resize=512, lambda_kl=0.001, iterations=None, epochs=None,
              save_diffs=True, device=torch.device('cuda:0')):
    fp = WideFeatures(resize=resize, device=device, save_in_memory=False, save_on_disk=False)
    data_loader = CustomDataLoader(fp, dataset, device=device, batch_size=8, shuffle=True, num_workers=8)
    if iterations is not None:
        epochs = (iterations - 1) // len(dataset) + 1

    vae = VAE().to(device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=0.0001, weight_decay=0.1)
    loss_fn = nn.functional.mse_loss
    for _ in tqdm(range(epochs)):
        for features, img, inv_mask, gt_mask, idx in data_loader:
            features = op_utils.scale_features(features)
            inv_mask = torch.nn.functional.interpolate((inv_mask[:, None]).to(device).float(), features.shape[-2:])
            val_mask = (inv_mask < 0.5).expand_as(features)

            f_hat = vae(features)
            features = torch.masked_select(features, val_mask)
            f_hat = torch.masked_select(f_hat, val_mask)
            mse = loss_fn(f_hat, features)
            loss = mse + vae.encoder.kl * lambda_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    Path(path).mkdir(parents=True, exist_ok=True)
    torch.save(vae.state_dict(), f"{path}/{dataset.object_name}.pt")

    if save_diffs:
        generate_vae_diffs(path, dataset, device, data_loader, resize=resize)


def generate_vae_diffs(vae_path, dataset, device, data_loader=None, resize=None):
    vae = VAE().to(device)
    vae.load_state_dict(torch.load(f"{vae_path}/{dataset.object_name}.pt", weights_only=True))
    cache_path = vae_path / dataset.object_name
    cache_path.mkdir(exist_ok=True, parents=True)
    (vae_path / dataset.object_name / "vis").mkdir(exist_ok=True, parents=True)

    if data_loader is None:
        fp = WideFeatures(resize=resize, device=device, save_in_memory=False, save_on_disk=False)
        data_loader = CustomDataLoader(fp, dataset, device=device, batch_size=8, shuffle=True)

    for features, img, inv_mask, gt_mask, idxs in data_loader:
        with torch.no_grad():
            features = op_utils.scale_features(features)

            f_hat = vae.inference(features)
            diffs = (features - f_hat)
        for diff, idx in zip(diffs, idxs):
            obj_name, class_name, img_name = dataset.path_tokens(dataset.img_paths[idx])
            file_path = cache_path / f"{class_name}_{img_name.split('.')[0]}.pt"
            torch.save(diff.clone(), file_path)
            vd = diff.abs().mean(dim=0, keepdim=True)
            vd = (vd - vd.min()) / (vd.max() - vd.min() + 1e-8)
            transforms.ToPILImage()(vd).save(vae_path / dataset.object_name / "vis" / f"{class_name}_{img_name.split('.')[0]}.jpg")

def main():
    parser = argparse.ArgumentParser(description="Train a VAE on either a SimpleDataset or an MVTec object.")

    parser.add_argument("dataset", choices=["simple", "mvtec"], help="Dataset type to use.")
    parser.add_argument("name", help=(
            "For 'simple': the dataset folder inside datasets/. "
            "For 'mvtec': the object name."
        ),
    )
    parser.add_argument("--resize", type=int, default=512, help="Image resize dimension (default: 512).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()

    op_utils.set_seed(args.seed)

    if args.dataset == "simple":
        dataset = SimpleDataset(data_root=f"datasets/{args.name}", name=args.name, resize=args.resize)
    else:
        dataset = MVTecDataset(data_root="datasets/mvtec_anomaly_detection", object_name=args.name)

    print(f"Training VAE on {dataset.name} located at {dataset.data_root}, for object {dataset.object_name}")
    train_vae(dataset, Path("cache") / dataset.name / "VAE", resize=args.resize, iterations=2000, save_diffs=True)


if __name__ == '__main__':
    main()

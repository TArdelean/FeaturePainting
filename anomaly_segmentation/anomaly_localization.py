import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import op_utils
from data import CustomDataLoader
from fca_add import FCAWithMask


@hydra.main(version_base=None, config_path="conf", config_name="localization")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    save_alpha_unit = cfg.save_alpha_unit
    save_alpha_tiff = cfg.save_alpha_tiff

    device = torch.device(cfg.device)
    dataset = instantiate(cfg.dataset)
    fp = instantiate(cfg.features)

    stat = FCAWithMask((7, 7), sigma_p=3.0, k_s=5, sigma_s=1.0, chunk_size=20)
    data_loader = CustomDataLoader(fp, dataset, batch_size=1)

    tiff_alphas = []
    vis_alphas = []
    for f_wide, img, inv_mask, gt_mask, idx in tqdm(data_loader):
        with torch.no_grad():
            inv_mask = torch.nn.functional.interpolate((inv_mask[:, None]).to(device).float(), (64, 64))
            val_mask = 1 - torch.nn.functional.conv2d(inv_mask, inv_mask.new_ones(1, 1, 3, 3), padding=1)[0, 0]
        alpha = stat(f_wide, val_mask)
        alpha = (val_mask > 0.5) * alpha
        tiff_alphas.append(alpha.clone())

        alpha = (alpha / alpha.max()).clip(0, 1)
        vis_alphas.append(alpha)
    if save_alpha_unit:
        op_utils.save_alpha_unit(vis_alphas, dataset, vis_size=cfg.vis_size)
    if save_alpha_tiff:
        op_utils.save_alpha_tiff(tiff_alphas, dataset)

    print(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    my_app()

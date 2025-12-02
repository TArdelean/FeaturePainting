import random
from pathlib import Path

import hydra
import numpy as np
import skimage
import skimage.morphology
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from skimage.filters import threshold_otsu
from sklearn import cluster

import op_utils
from data.feature_provider import FeatureProvider
from data import MaskMemory


def mod_otsu(sample, b, border):
    if border > 0:
        heat = sample[..., border:-border, border:-border]
    else:
        heat = sample
    return threshold_otsu(heat ** b) ** (1 / b)

def inh_array(arr, dtype=None):
    try:
        return np.array(arr, dtype=dtype)
    except Exception:
        return np.array(arr, dtype=object)

def compute_global_threshold(dataset, memory, border, beta):
    per_im_thr = []
    for i in range(len(dataset)):
        sample = memory[i, 0].numpy()
        thr = mod_otsu(sample, beta, border)
        per_im_thr.append(thr)
    return threshold_otsu(np.array(per_im_thr))

def compute_groups_descriptors(features, group_map, mask, blur_f=1.0, blur_m=None, tau=1.0):
    features = op_utils.blur(features, sigma=blur_f)
    if blur_m is not None:
        mask = op_utils.blur(mask, sigma=blur_m)
    f_good = features - torch.mean(features, dim=(-2, -1), keepdim=True)
    mask = mask / tau
    f_groups = []
    weights = []
    for label in range(int(group_map.max()) + 1):
        idxs = group_map == label
        f_crt = f_good[:, idxs]
        w_group = torch.softmax(mask[idxs], dim=-1)[None]
        f_group = (f_crt * w_group).sum(dim=1)
        f_groups.append(f_group)
        weights.append(mask[idxs].max())
    return torch.stack(f_groups, dim=0), features.new_tensor(weights)

def sample_antagonists(pools, labels, cnt, n_clusters, seed=42):
    # Stratified sampling of negative contrastive pairs (antagonists) from each pool -- up to cnt elements
    # pool: B x E
    # labels: B
    # cnt: int
    # Returns: B x cnt
    op_utils.set_seed(seed)
    antagonists = []
    for pool in pools:
        candidate = [[] for _ in range(n_clusters)]
        shuffled = pool[torch.randperm(len(pool))]
        for item in shuffled:
            candidate[labels[item]].append(item.item())
        # Should shuffle candidates
        random.shuffle(candidate)
        free = [pool.new_tensor([])]
        reminder = cnt
        while len(candidate) != 0:
            per_group = reminder // len(candidate)
            free_now = [pool.new_tensor(cand) for cand in candidate if len(cand) <= per_group]
            if len(free_now) == 0:
                break
            free.extend(free_now)
            reminder -= sum(map(len, free_now))
            candidate = [cand for cand in candidate if len(cand) > per_group]
        if len(candidate) != 0:
            free = torch.cat(free, dim=0)
            other = torch.stack([pool.new_tensor(cand)[:per_group+1] for cand in candidate])
            other_selection = other.T.flatten()[:reminder]
            antagonist = torch.cat([free, other_selection], dim=0)
            if antagonist.shape[0] != cnt:
                print(free.shape, other.shape, [len(cand) for cand in candidate])
                raise Exception("Error")
        else:
            antagonist = torch.cat(free, dim=0)
        antagonists.append(antagonist)
    return torch.stack(antagonists, dim=0)

def remap_indices(cum_indices, lengths):
    cum_lengths = torch.cumsum(cum_indices.new_tensor([0, *lengths]), dim=-1)
    group_indices = torch.searchsorted(cum_lengths, cum_indices, right=True) - 1
    local_idx = cum_indices - cum_lengths[group_indices]
    return torch.stack([group_indices, local_idx], dim=-1)

def create_groups_and_friends(tiff_dir, dataset, fp, f_cnt=8, e_cnt='auto', e_pool='auto', tau=1.0, beta=1.5, border=5,
                              erosion=(2, 2), min_anomaly_size=12):
    all_groups = []
    all_descriptors = []
    all_weights = []
    masks = MaskMemory(tiff_dir, dataset, 'cpu', is_tiff=True)
    masks.memory = op_utils.clear_borders(masks.memory, border=border)
    global_thr = compute_global_threshold(dataset, masks.memory, border, beta)
    print("Printing the size of the identified anomalous connected components:")
    for i in range(len(dataset)):
        sample = masks.memory[i, 0].numpy()
        thr = mod_otsu(sample, beta, border)
        thr = max(thr, global_thr)

        binary = sample > thr
        binary = skimage.morphology.binary_erosion(binary, footprint=np.ones(erosion))
        binary = skimage.morphology.remove_small_objects(binary, min_size=min_anomaly_size)

        labeled_image, count = skimage.measure.label(binary, return_num=True)
        labeled_image[sample < 1e-6] = -1
        groups = [np.asarray(np.where(labeled_image == ind)).T for ind in range(count + 1)]
        all_groups.append(inh_array(groups))
        print(dataset.img_paths[i].stem, count, list(map(len, groups)))

        g_descriptors, g_weights = compute_groups_descriptors(fp.get(i), torch.tensor(labeled_image, device=fp.device),
                                                              torch.tensor(sample, device=fp.device), tau=tau)
        all_descriptors.append(g_descriptors)
        all_weights.append(g_weights)

    lengths = [g_descriptors.shape[0] for g_descriptors in all_descriptors]
    flat_descriptors = torch.cat(all_descriptors, 0)
    flat_weights = torch.cat(all_weights, 0)
    dists = torch.square(flat_descriptors[:, None, :] - flat_descriptors[None, :, :]).mean(dim=-1)  # B x B

    top_indices = torch.argsort(dists, dim=1)
    friends = top_indices[:, 1:f_cnt + 1]  # B x F
    friends = remap_indices(friends.reshape(-1), lengths).reshape(*friends.shape, 2)  # B x F x 2
    friends = [_.cpu().numpy() for _ in torch.split(friends, lengths)]

    # Preliminary clustering for stratified sampling
    cm = cluster.KMeans(n_clusters=dataset.n_clusters, random_state=42, n_init=10)
    cm.fit(flat_descriptors.cpu().numpy())

    print("Number of regions in each preliminary cluster: ", np.bincount(cm.labels_))
    if e_pool == 'auto':
        e_pool = np.bincount(cm.labels_).sum() - np.bincount(cm.labels_).max(initial=0)
        e_cnt = e_pool
    antagonists = top_indices[:, -e_pool:]  # B x E
    a_weights = flat_weights[antagonists]  # B x E
    antagonists = torch.gather(antagonists, index=torch.argsort(a_weights, descending=True, dim=-1), dim=-1)  # B x E
    antagonists = sample_antagonists(antagonists, cm.labels_, e_cnt, dataset.n_clusters)
    antagonists = antagonists[:, :e_cnt]
    antagonists = remap_indices(antagonists.reshape(-1), lengths).reshape(*antagonists.shape, 2)  # B x E x 2
    antagonists = [_.cpu().numpy() for _ in torch.split(antagonists, lengths)]

    return all_groups, friends, antagonists


@hydra.main(version_base=None, config_path="conf", config_name="create_cld")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    op_utils.set_seed(42)

    dataset = instantiate(cfg.dataset)
    dataset.n_clusters = cfg.n_clusters
    fp: FeatureProvider = instantiate(cfg.features).init(dataset)
    tiff_dir = Path(cfg.masks_root)
    save_path = Path(cfg.pairs_dataset_path)

    all_groups, friends, antagonists = create_groups_and_friends(tiff_dir, dataset, fp,
                                                                 beta=cfg.beta, border=cfg.border, f_cnt=cfg.f_cnt)

    save_path.mkdir(exist_ok=True, parents=True)
    np.save(str(save_path / 'groups.npy'), np.asarray(all_groups, dtype="object"))
    np.save(str(save_path / 'friends.npy'), np.asarray(friends, dtype="object"))
    np.save(str(save_path / 'antagonists.npy'), np.asarray(antagonists, dtype="object"))
    with open(str(save_path / 'groups.check.txt'), "w") as text_file:
        text_file.write(str(np.asarray(all_groups, dtype="object")))


if __name__ == '__main__':
    my_app()

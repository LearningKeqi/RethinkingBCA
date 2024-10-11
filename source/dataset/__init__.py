from omegaconf import DictConfig, open_dict
from .abcd import load_abcd_data
from .abide import load_abide_data
from .pnc import load_pnc_data
from .hcp import load_hcp_data
from .dataloader import init_dataloader, init_stratified_dataloader
from typing import List
import torch.utils as utils
import torch
from .process_node_feature import preprocess_nodefeature


def dataset_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:

    assert cfg.dataset.name in ['abcd', 'abide', 'pnc', 'hcp']

    load_name = cfg.dataset.name

    # print('dataset 0')
    
    datasets = eval(
        f"load_{load_name}_data")(cfg)

    dataloaders = init_stratified_dataloader(cfg, *datasets) \
        if cfg.dataset.stratified \
        else init_dataloader(cfg, *datasets)

    return dataloaders

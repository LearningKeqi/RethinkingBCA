from operator import mod
from .training import Train
from .BrainGNN_orig_training import BrainGNN_orig_Train
from .Brainnetcnntraining import Brainnetcnn_Train
from .MLPtraining import MLP_Train
from .Neurographtraining import Neurograph_Train
from .BrainGBtraining import BrainGB_Train
from omegaconf import DictConfig
from typing import List
import torch
from source.components import LRScheduler
import logging
import torch.utils.data as utils



def training_factory(config: DictConfig,
                     model: torch.nn.Module,
                     optimizers: List[torch.optim.Optimizer],
                     lr_schedulers: List[LRScheduler],
                     dataloaders: List[utils.DataLoader],
                     logger: logging.Logger) -> Train:

    train = config.model.get("train", None)
    if not train:
        train = config.training.name
    
    print(f'train={train}')
    
    return eval(train)(cfg=config,
                       model=model,
                       optimizers=optimizers,
                       lr_schedulers=lr_schedulers,
                       dataloaders=dataloaders,
                       logger=logger)

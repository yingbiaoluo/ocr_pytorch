import copy
import torch
from torch.utils.data import DataLoader


class Quest_CollateFN():
    def __init__(self):
        pass

    def __call__(self, batch):
        img, label = zip(*batch)
        ls = []
        for l in label:
            ls.append(l)
        return torch.stack(img, 0), ls


def get_dataloader(config_dataset):
    from . import dataset
    config = copy.deepcopy(config_dataset)

    if config['loader']['collate_fn']:
        config['loader']['collate_fn'] = eval(config['loader']['collate_fn'])()
    else:
        config['loader']['collate_fn'] = None

    _dataset = dataset.Quest_dataset(**config['dataset']['args'])
    loader = DataLoader(dataset=_dataset, **config['loader'])
    return loader

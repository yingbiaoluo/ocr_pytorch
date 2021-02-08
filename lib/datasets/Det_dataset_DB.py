import os
import json
import cv2
import copy
from tqdm import tqdm

from torch.utils.data import Dataset


class JsonDataset(Dataset):
    def __init__(self, cfg):
        assert cfg.img_mode in ['RGB', 'BGR', 'GRAY']
        self.ignore_tags = cfg.ignore_tags

        self.data_list = self

    def load_data(self, path: str) -> list:
        return 0

    def __getitem__(self, index):
        return 0

    def __len__(self):
        return 0


if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader
    from config.det_train_db_config import config
    from pprint import pprint

    pprint(config)

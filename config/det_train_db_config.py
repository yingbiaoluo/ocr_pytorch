from addict import Dict

config = Dict()
config.exp_name = 'DBNet'

config.train_options = {
    'resume_from': '',  # resume most recent training
    'checkpoint_save_dir': f'./output/{config.exp_name}/checkpoint',
    'device': 'cuda:0',
    'epochs': 300,
}

config.SEED = 927
config.optimizer = {
    'type': 'adam',
    'lr': 0.001,
    'weight_decay': 1e-4,
}

config.model = {
    'type': 'DetModel',
    # 'backbone': {'type': 'ResNet', 'layers': 18, 'pretrained': False},
    'backbone': {'type': 'MobileNetV3', 'scale': 0.5, 'model_name': 'large', 'pretrained': False},
    'neck': {'type': 'FPN', 'inner_channels': 256},
    'head': {'type': 'DBHead'},
    'in_channels': 3,
}

config.loss = {
    'type': 'DBLoss',
    'alpha': 1,
    'beta': 10,
}

config.post_process = {
    'type': 'DBPostProcess',
    'thresh': 0.3,  # 二值化输出map的阈值
    'box_thresh': 0.7,  # 低于此阈值的box丢弃
    'unclip_ratio': 1.5,  # 扩大框的比例
}

config.dataset = {
    'train': {
        'dataset': {
            'type': 'JsonDataset',
            'file': r'/home/lyb/',
            'ignore_tags': ['*', '###'],
            'img_mode': 'RGB'
        },
        'loader': {

        }
    },
    'eval': {
        'dataset': {
            'type': 'JsonDataset',
            'ignore_tags': ['*', '###'],
            'img_mode': 'RGB'
        },
        'loader': {

        }
    }
}
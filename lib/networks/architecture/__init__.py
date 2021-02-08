from addict import Dict
import copy
from .Det_model import DetModel
from .Rec_model import RecModel

support_model = ['DetModel', 'RecModel']


def build_model(config):
    copy_cfg = copy.deepcopy(config)
    arch_type = copy_cfg.pop('type')
    assert arch_type in support_model, f'{arch_type} is not developed yet!, only {support_model} are support now'
    arch_model = eval(arch_type)(Dict(copy_cfg))
    return arch_model

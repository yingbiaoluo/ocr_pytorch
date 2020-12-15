from . import model, Loss


def get_model(config, device='cpu'):
    _model = getattr(model, config['type'])(config['args'], device=device).to(device)
    return _model


def get_loss(config):
    return getattr(Loss, config['type'])(**config['args'])
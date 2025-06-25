from torch.optim import SGD, Adam, AdamW

def initialize_optimizer(config, model):
    # initialize optimizers
    if config.optimizer=='SGD':
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = SGD(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum
        )
    elif config.optimizer == 'AdamW':
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = AdamW(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f'Optimizer {config.optimizer} not recognized.')

    return optimizer
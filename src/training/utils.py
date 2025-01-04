import torch
import torch.nn as nn
import torch.optim as optim

def get_optimizer(model, config):
    """
    Get optimizer based on config
    """
    if config['optimizer']['name'] == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=config['optimizer']['learning_rate'],
            weight_decay=config['optimizer']['weight_decay']
        )
    elif config['optimizer']['name'] == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=config['optimizer']['learning_rate'],
            momentum=config['optimizer']['momentum'],
            weight_decay=config['optimizer']['weight_decay']
        )
    else:
        raise ValueError(f"Optimizer {config['optimizer']['name']} not supported")

def get_scheduler(optimizer, config):
    """
    Get learning rate scheduler based on config
    """
    if config['scheduler']['name'] == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['scheduler']['factor'],
            patience=config['scheduler']['patience'],
            verbose=True
        )
    elif config['scheduler']['name'] == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['scheduler']['min_lr']
        )
    else:
        raise ValueError(f"Scheduler {config['scheduler']['name']} not supported")

def get_criterion(config, class_weights=None):
    """
    Get loss function based on config
    """
    if config['training']['criterion'] == 'cross_entropy':
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights.to(config['device']))
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Criterion {config['training']['criterion']} not supported")
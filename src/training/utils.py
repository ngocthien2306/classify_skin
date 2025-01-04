import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

class MarginLoss(nn.Module):
    """
    Margin Loss for Capsule Networks as described in Dynamic Routing Between Capsules paper
    """
    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        """
        Args:
            m_plus: The margin for positive cases (default: 0.9)
            m_minus: The margin for negative cases (default: 0.1) 
            lambda_: Down-weighting of the loss for absent classes (default: 0.5)
        """
        super(MarginLoss, self).__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_

    def forward(self, outputs, targets, num_classes=7):
        """
        Args:
            outputs: Network outputs after squashing (class probabilities)
            targets: Ground truth labels
            num_classes: Number of classes (default: 7 for HAM10000)

        Returns:
            Total loss value
        """
        # Convert targets to one-hot encoding
        target_oh = F.one_hot(targets, num_classes=num_classes).float()
        
        # Compute losses for each dimension
        present_error = F.relu(self.m_plus - outputs) ** 2  # for positive cases
        absent_error = F.relu(outputs - self.m_minus) ** 2  # for negative cases
        
        # Combine losses with class weighting
        loss = target_oh * present_error + \
               self.lambda_ * (1.0 - target_oh) * absent_error
        
        # Sum over class dimension, mean over batch
        loss = loss.sum(dim=1).mean()
        
        return loss

class CombinedLoss(nn.Module):
    """
    Combined loss that can mix multiple loss functions with weights
    """
    def __init__(self, margin_weight=1.0, ce_weight=0.0, class_weights=None, 
                 m_plus=0.9, m_minus=0.1, lambda_=0.5):
        """
        Args:
            margin_weight: Weight for margin loss
            ce_weight: Weight for cross entropy loss
            class_weights: Class weights for cross entropy loss
            m_plus: Margin loss parameter
            m_minus: Margin loss parameter
            lambda_: Margin loss parameter
        """
        super(CombinedLoss, self).__init__()
        self.margin_loss = MarginLoss(m_plus, m_minus, lambda_)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
        self.margin_weight = margin_weight
        self.ce_weight = ce_weight

    def forward(self, outputs, targets, num_classes=7):
        """
        Args:
            outputs: Network outputs
            targets: Ground truth labels
            num_classes: Number of classes
        """
        loss = 0
        if self.margin_weight > 0:
            loss += self.margin_weight * self.margin_loss(outputs, targets, num_classes)
        if self.ce_weight > 0:
            loss += self.ce_weight * self.ce_loss(outputs, targets)
        return loss
    
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
    
    Args:
        config: Configuration dictionary
        class_weights: Optional class weights tensor for weighted loss
        
    Returns:
        Loss function
    """
    criterion_name = config['training']['criterion']
    
    if criterion_name == 'cross_entropy':
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights.to(config['device']))
        return nn.CrossEntropyLoss()
        
    elif criterion_name == 'margin':
        margin_config = config['training'].get('margin_loss', {})
        return MarginLoss(
            m_plus=margin_config.get('m_plus', 0.9),
            m_minus=margin_config.get('m_minus', 0.1),
            lambda_=margin_config.get('lambda', 0.5)
        )
        
    elif criterion_name == 'combined':
        combined_config = config['training'].get('combined_loss', {})
        return CombinedLoss(
            margin_weight=combined_config.get('margin_weight', 1.0),
            ce_weight=combined_config.get('ce_weight', 0.0),
            class_weights=class_weights,
            m_plus=combined_config.get('m_plus', 0.9),
            m_minus=combined_config.get('m_minus', 0.1),
            lambda_=combined_config.get('lambda', 0.5)
        )
        
    else:
        raise ValueError(f"Criterion {criterion_name} not supported")
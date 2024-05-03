from ..loss.traversability_loss import TraversabilityLoss
from ..loss.standard_loss import StandardLoss
from ..loss.standard_loss_unlabelled_neg import StandardLossUnlabelledNeg
from ..loss.confidence_loss import ConfidenceLoss
from ..loss.traversability_loss_L1 import TraversabilityLossL1
def loss_builder(config):
    if config['loss']['loss'] == 'TraversabilityLoss':
        return TraversabilityLoss(config)
    elif config['loss']['loss'] == 'StandardLoss':
        return StandardLoss(config)
    elif config['loss']['loss'] == 'StandardLossUnlabelledNeg':
        return StandardLossUnlabelledNeg(config)
    elif config['loss']['loss'] == 'ConfidenceLoss':
        return ConfidenceLoss(config)
    elif config['loss']['loss'] == 'TraversabilityLossL1':
        return TraversabilityLossL1(config)
    else:
        raise ValueError(f"Loss function {config['loss']['loss']} not implemented")
    

from ..loss.traversability_loss import TraversabilityLoss
from ..loss.standard_loss import StandardLoss
from ..loss.standard_loss_unlabelled_neg import StandardLossUnlabelledNeg
from ..loss.confidence_loss import ConfidenceLoss
from ..loss.traversability_loss_L1 import TraversabilityLossL1
def loss_builder(config,rank):
    if config['loss']['loss'] == 'TraversabilityLoss':
        return TraversabilityLoss(config,rank)
    elif config['loss']['loss'] == 'StandardLoss':
        return StandardLoss(config,rank)
    elif config['loss']['loss'] == 'StandardLossUnlabelledNeg':
        return StandardLossUnlabelledNeg(config,rank)
    elif config['loss']['loss'] == 'ConfidenceLoss':
        return ConfidenceLoss(config,rank)
    elif config['loss']['loss'] == 'TraversabilityLossL1':
        return TraversabilityLossL1(config,rank)
    else:
        raise ValueError(f"Loss function {config['loss']['loss']} not implemented")
    

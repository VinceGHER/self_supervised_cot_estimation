import torch

class MAEMetric:
    def __init__(self):
        """
        Initialize states required for MAE computation.
        """
        self.reset()

    def reset(self):
        """
        Resets the state for a new set of MAE computations.
        """
        self.total_error = 0.0
        self.total_count = 0

    def update(self, preds, target):
        """
        Update the running MAE calculation with a new set of predictions and targets.
        
        Args:
            preds (torch.Tensor): The model predictions.
            target (torch.Tensor): The ground truth values.
        """
        # Compute the absolute error
        abs_error = torch.abs(preds - target)

        # Update total error and count
        self.total_error += torch.sum(abs_error).item()
        self.total_count += target.numel()

    def compute(self):
        """
        Compute the current MAE based on accumulated states.
        
        Returns:
            float: The computed MAE.
        """
        if self.total_count == 0:
            raise RuntimeError("MAE computation called without any update. Ensure data was passed.")
        return self.total_error / self.total_count

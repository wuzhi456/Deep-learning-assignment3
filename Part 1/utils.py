import torch


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count

    def __str__(self):
        fmtstr = '{name}: {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


@torch.no_grad()
def accuracy(output, target):
    """
    Compute accuracy for predictions.
    
    Args:
        output: Model predictions (logits), shape (batch_size, num_classes)
        target: Ground truth labels, shape (batch_size,)
    
    Returns:
        Accuracy as a percentage (0-100)
    """
    # Get predicted class (index with maximum value)
    predictions = torch.argmax(output, dim=1)
    # Compare predictions with targets
    correct = (predictions == target).float()
    # Return accuracy as percentage
    return correct.mean().item() * 100

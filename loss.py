import torch, torchvision
import torch.nn.functional as F

################## Custom losses ##################
# Just MVLoss will be used in final thesis: the others have been tested for classification, but have not given good results

class AlphaLoss(torch.nn.Module):
    def __init__(self, alpha=1.0):
        """ This loss calculates the square difference between predicted age and groundtruth; this is multiplied by 
        values that are 'responsible' for the output and from those ones shall be backpropagated the gradient."""
        super().__init__()
        self.alpha = alpha
    
    def forward(self, batch, targets):
        indices = torch.argmax(batch, -1).float()
        values = torch.nn.functional.cross_entropy(batch, targets.long(), reduction='none')
        # normalize the difference through the vector of outputs' size (that is the number of classes)
        tmp = values * (1 + self.alpha * torch.abs((indices - targets) / batch.shape[1]))
        return torch.mean(tmp)
        

    def __repr__(self):
        return self.__class__.__name__ + "(alpha = {})".format(self.alpha)


class AlphaLossWithoutNorm(torch.nn.Module):
    def __init__(self, alpha=1.0):
        """ This loss calculates the square difference between predicted age and groundtruth; this is multiplied by 
        values that are 'responsible' for the output and from those ones shall be backpropagated the gradient."""
        super().__init__()
        self.alpha = alpha
    
    def forward(self, batch, targets):
        indices = torch.argmax(batch, -1).float()
        values = torch.nn.functional.cross_entropy(batch, targets.long(), reduction='none')
        # normalize the difference through the vector of outputs' size (that is the number of classes)
        tmp = values * (1 + self.alpha * torch.abs((indices - targets)))
        return torch.mean(tmp)
        

    def __repr__(self):
        return self.__class__.__name__ + "(alpha = {})".format(self.alpha)


class SqAlphaLoss(torch.nn.Module):
    def __init__(self, alpha=1.0):
        """ This loss calculates the square difference between predicted age and groundtruth; this is multiplied by 
        values that are 'responsible' for the output and from those ones shall be backpropagated the gradient."""
        super().__init__()
        self.alpha = alpha
    
    def forward(self, batch, targets):
        indices = torch.argmax(batch, -1).float()
        values = torch.nn.functional.cross_entropy(batch, targets.long(), reduction='none')
        # normalize the difference through the vector of outputs' size (that is the number of classes)
        tmp = values + self.alpha * torch.abs((indices - targets))**2
        return torch.mean(tmp)
        

    def __repr__(self):
        return self.__class__.__name__ + "(alpha = {})".format(self.alpha)


class MVLoss(torch.nn.Module):

    def __init__(self, l1=0.2, l2=0.05):
        # l1, l2 normalization parameters for mean and variance loss (respectively)
        super().__init__()
        self.l1 = l1
        self.l2 = l2
    
    def forward(self, batch:torch.Tensor, targets):
        targets=targets.long()
        probs = torch.nn.functional.softmax(batch, dim=1).cpu()
        ages = torch.arange(batch.shape[1])
        mean = torch.squeeze((probs * ages).sum(1, keepdim=True), dim=1)
        mse = (mean - targets)**2
        mean_loss = mse.mean() / 2.0

        # variance loss
        var = (ages[None, :] - mean[:, None])**2
        variance_loss = (probs * var).sum(1, keepdim=True).mean()
        func_loss = torch.nn.functional.cross_entropy(batch, targets)

        # returned as tuple to allow single-value printing/visualization
        return func_loss, self.l1 * mean_loss, self.l2 * variance_loss
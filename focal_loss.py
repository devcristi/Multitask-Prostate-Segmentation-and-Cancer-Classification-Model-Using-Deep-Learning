import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss pentru clasificare binară.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        :param alpha: factor de echilibrare a claselor (0 < alpha <= 1).
                      De obicei, alpha < 0.5 pentru a penaliza mai tare clasa pozitivă (minoritară).
        :param gamma: exponentul de focalizare (cu cât e mai mare, cu atât
                      penalizezi mai tare exemplele dificil de clasificat).
        :param reduction: 'none', 'mean' sau 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [B], sau [B, 1] -> logit pentru clasa pozitivă
        targets: [B], valori 0 sau 1
        """
        # Asigură-te că logits e de forma [B]
        if logits.ndim > 1:
            logits = logits.view(-1)
        # la fel și pentru targets
        targets = targets.view(-1)

        # Transformare logit -> probabilitate p = sigmoid(logit)
        p = torch.sigmoid(logits)
        # p_t = p dacă y=1, altfel 1-p
        p_t = p * targets + (1 - p) * (1 - targets)

        # factorul alpha_t: alpha dacă y=1, altfel (1 - alpha)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # focal weight = (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # log(p_t)
        log_p_t = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )  # BCE pe fiecare eșantion

        # focal loss = alpha_t * focal_weight * log(p_t)
        loss = alpha_t * focal_weight * log_p_t

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

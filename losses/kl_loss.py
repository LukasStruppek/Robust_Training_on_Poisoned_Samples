import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch


class KDLoss(_Loss):
    """
    Knowledge distillation loss function
    """

    def __init__(self, temperature, alpha, reduction='mean'):
        """
        Args:
            temperature: Temperature used in softmax
            alpha: Weighting factor for distillation loss
            reduction: Reduction method for loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input_student, input_teacher, target):
        """
        Args:
            input_student: Output of student model
            input_teacher: Output of teacher model
            target: Ground truth labels
        """
        input_student = F.log_softmax(input_student / self.temperature, dim=1)
        input_teacher = F.softmax(input_teacher / self.temperature, dim=1)
        distillation_loss = F.kl_div(
            input_student, input_teacher,
            reduction='batchmean') * (self.temperature**2)
        student_loss = F.cross_entropy(input_student,
                                       target,
                                       reduction=self.reduction)
        loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        return loss
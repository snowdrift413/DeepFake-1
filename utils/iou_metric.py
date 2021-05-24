## Compute the Intersection over Union(IoU) per class and corresponding mean(mIoU) for evaluating semantic segmentation
import torch
import numpy as np
from fsgan.utils.confusion_matrix import ConfusionMatrix

class IoUMetric(object):
    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized)

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index, )
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("ignore_index must be and int or iterable")

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        assert predicted.size(0) == target.size(0), \
            '예측값과 타겟값이 일치하지 않는다. number of targets and predicted do not match'
        assert predicted.dim() == 3 or predicted.dim() == 4, \
            '예측값의 차원은 3 또는 4이어야 한다. predictions must be of dimension (N, H, W) or (N, K, H, W)'
        assert target.dim() == 3 or target.dim() == 4, \
            '타겟값의 차원 3 또는 4이어야 한다. targets must be of dimension (N, H, W) or (N, K, H, W)'
        
        # if the tensor is in categorical format convert it to integer format
        if predicted.dim() == 4:
            _, predicted = predicted.max(1)
        if target.dim() == 4:
            _, target = target.max(1)

        self.conf_metric.add(predicted.view(-1), target.view(-1))

    # compute the IoU and mIoU
    def value(self):
        conf_matrix = self.conf_metric.value()

        if self.ignore_index is not None:
            for index in self.ignore_index:
                conf_matrix[:, self.ignore_index] = 0
                conf_matrix[self.ignore_index, :] = 0
        
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # just in case we got a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
        
        return iou, np.nanmean(iou)
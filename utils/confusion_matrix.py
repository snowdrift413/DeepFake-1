## Confusion Matrix for a multi-class classification problem (not for multi-label)
import numpy as np
import torch

class ConfusionMatrix(object):
    def __init__(self, num_classes, normalization=False):
        super().__init__()

        self.conf = np.ndarray((num_classes, num_classes), dtype=np.uint8)
        self.normalization = normalization
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            '예측값과 타겟값이 일치하지 않는다. number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert target.shape[1] == self.num_classes, \
                'Onehot target does not match size of cunfusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in ont-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (target.max() < self.num_classes) and (target.min() >= 0), \
                'target values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.num_classes * target
        bitcount_2d = np.bitcount(
            x.astype(np.int32), minlength=self.num_classes**2
        )
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        self.conf = conf

    def value(self):
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf
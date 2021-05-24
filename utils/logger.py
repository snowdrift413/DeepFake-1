## Tensorboard Logger
from tensorboardX import SummaryWriter

## Computes and stores the average and current value
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class TensorboardLogger(SummaryWriter):
    def __init__(self, log_dir=None):
        super(TensorboardLogger, self).__init__(log_dir)
        self._tb_logger = SummaryWriter(log_dir) if log_dir is not None else None
        self.log_dict = {}

    def reset(self, prefix=None):
        self.prefix = prefix
        self.log_dict.clear()

    def update(self, category='losses', **kwargs):
        if category not in self.log_dict:
            self.log_dict[category] = {}
        category_dict = self.log_dict[category]

        for key, val in kwargs.items():
            if key not in category_dict:
                category_dict[key] = AverageMeter()
            category_dict[key].update(val)

    def log_scalars_cal(self, main_tag, global_step=None):
        pass
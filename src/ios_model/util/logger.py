import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.reset()
        self.name = name

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


class EpochLogger(object):
    def __init__(self, name_list):
        self.logger_dict = {name: AverageMeter(name) for name in name_list}
        self.reset()

    def reset(self):
        for v in self.logger_dict.values():
            v.reset()

    def set_batch_log(self, names, values, numbers):
        for name, val, n in zip(names, values, numbers):
            self.logger_dict[name].update(val, n)

    def get_batch_log(self):
        return {k: v.val for k, v in self.logger_dict.items()}

    def get_epoch_log(self):
        return {k: v.avg for k, v in self.logger_dict.items()}


def create_logger(model_name, checkpoint_root, copy_file_list):
    now = datetime.now()
    checkpoint_folder = os.path.join(
        checkpoint_root, "%s_%s" % (model_name, str(now.strftime("%Y%m%d_%H-%M-%S")))
    )
    if os.path.exists(checkpoint_folder) == False:
        os.system("mkdir %s" % checkpoint_folder)
        os.system(f"mkdir {checkpoint_folder}/logging")
        os.system(f"mkdir {checkpoint_folder}/weight")
        for path in copy_file_list:
            os.system(f"cp -r {path} {checkpoint_folder}")
    writer = SummaryWriter(log_dir=f"{checkpoint_folder}/logging")
    return writer, checkpoint_folder

import paddle

from reid.utils.utils import Timer
from reid.loss import TriHardPlusLoss


class Trainer(object):
    def __init__(self):
        self.timer = Timer()

        self.pid_criterion = paddle.nn.CrossEntropyLoss()
        self.triplet_criterion = TriHardPlusLoss(0.0)

        self.trainables = []
        self.untrainables = []

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def before_train(self, *args, **kwargs):
        for trainable in self.trainables:
            trainable.train()
            trainable.stop_gradient = False

        for untrainable in self.untrainables:
            untrainable.eval()
            untrainable.stop_gradient = True

        self.timer.reset()

    def train_step(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_loss(self, *args, **kwargs):
        raise NotImplementedError

    def after_train(self, *args, **kwargs):
        for trainable in self.trainables:
            trainable.eval()
            trainable.stop_gradient = True

        for untrainable in self.untrainables:
            untrainable.eval()
            untrainable.stop_gradient = True

    def basic_criterion(self, pooled, preds, pids):
        pid_loss = self.pid_criterion(preds, pids)
        triplet_loss, *_ = self.triplet_criterion(pooled, pids)
        return pid_loss, triplet_loss

    def _parse_data(self, inputs):
        imgs, _, pids, _, *_ = inputs
        return imgs, pids

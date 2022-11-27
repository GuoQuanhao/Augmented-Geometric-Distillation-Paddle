# -*- coding: utf-8 -*-
# Time    : 2020/2/10 10:03
# Author  : Yichen Lu

from math import inf
import paddle
from reid.models.backbone.resnet import ResNet
from paddle.optimizer.lr import LRScheduler

# step function should be called before picking off epoch training.


class CombinedLRSchduler(object):
    def __init__(self, schedulers):
        self.schedulers = schedulers

    def step(self, epoch):
        for scheduler in self.schedulers:
            scheduler.step(epoch)
        current_lrs = [scheduler.get_lr() for scheduler in self.schedulers]

        return current_lrs

    def get_lr(self):
        return [scheduler.get_lr() for scheduler in self.schedulers]

    def set_lr(self, lr):
        for scheduler in self.schedulers:
            scheduler.set_lr(lr)


class WarmupLRScheduler(LRScheduler):
    def __init__(self, learning_rate, warmup_epochs=10, base_lr=1e-2, milestones=(inf, ), start_epoch=1, verbose=False):
        self.start_epoch = start_epoch
        self.last_epoch = self.start_epoch
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.milestones = milestones
        assert self.milestones[0] > self.warmup_epochs, "First milestone epoch should be greater than warmup-epochs."
        super(WarmupLRScheduler, self).__init__(learning_rate, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs + 1:
            current_lr = self.base_lr * (self.last_epoch - self.start_epoch) / self.warmup_epochs
            if(current_lr < 0):
                return self.base_lr
        else:
            current_lr = self.base_lr
            for milestone in self.milestones:
                if self.last_epoch >= milestone + 1:
                    current_lr *= 0.1
        return current_lr


if __name__ == '__main__':
    model = ResNet()
    optimizer = paddle.optimizer.Momentum(model.parameters(), learning_rate=0.01)
    scheduler = WarmupLRScheduler(optimizer, milestones=(30, 50))
    for epoch in range(1, 61):
        scheduler.step(epoch)
        print(f"cuurent learning rate: {scheduler.get_lr()}")

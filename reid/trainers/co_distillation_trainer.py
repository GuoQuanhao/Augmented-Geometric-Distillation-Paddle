# -*- coding: utf-8 -*-
# Time    : 2021/8/10 9:47
# Author  : Yichen Lu

import paddle
from paddle.nn import functional as F

from reid.trainers import InversionIncrementalTrainer
from reid.loss import base
from reid.utils import AverageMeters, AverageMeter
from reid.loss.distillation import pairwise_residual


class CoDistillationTrainer(InversionIncrementalTrainer):
    def __init__(self,
                 networks_prec,
                 networks,
                 networks_prev,
                 incremental_classes,
                 optimizer,
                 lr_scheduler,
                 algo_config=None,
                 **kwargs
                 ):
        super(CoDistillationTrainer, self).__init__(networks_prec,
                                                    networks,
                                                    incremental_classes,
                                                    optimizer,
                                                    lr_scheduler,
                                                    algo_config,
                                                    )

        self.networks_prev = networks_prev
        self.untrainables += [self.networks_prev]

        self.co_distillation_config = algo_config["co_distillation"]
        self.domain_criterion = base.MMDLoss()

        self.meters = AverageMeters(AverageMeter("Batch Time"),
                                    AverageMeter("Pid Loss"),
                                    AverageMeter("Triplet Loss"),
                                    AverageMeter("Imp Distillation Loss"),
                                    AverageMeter("Inc Distillation Loss"),
                                    )

    def train(self, epoch, training_loader, training_loader_prec):
        self.before_train(epoch)

        training_loader_prec_iter = iter(training_loader_prec)

        for i, inputs in enumerate(training_loader):
            try:
                inputs_prec = next(training_loader_prec_iter)
            except StopIteration:
                training_loader_prec_iter = iter(training_loader_prec)
                inputs_prec = next(training_loader_prec_iter)

            inputs, pids = self._parse_data(inputs)
            inputs_prec, pids_prec = self._parse_data(inputs_prec)
            # To avoid overlapped pids
            pids_prec = pids_prec + self.incremental_classes
            losses = self.train_step(inputs, pids, inputs_prec, pids_prec)

            self.meters.update([self.timer(),
                                *losses,
                                ])

            print(f"Epoch: [{epoch}][{i + 1}/{len(training_loader)}], " + self.meters())

        self.after_train()

    def before_train(self, epoch):
        super(InversionIncrementalTrainer, self).before_train()
        self.lr_scheduler.step(epoch)

    def train_step(self, inputs, pids, inputs_prec, pids_prec):
        """
        New data + model -> outputs
        New data + preceding model -> outdated_outputs
        Old data + model -> outputs_prec
        Old data + preceding model -> outdated_outputs_prec
        New data + preview model -> outputs_prev
        """

        self.optimizer.zero_grad()
        outdated_outputs_prec = self.networks_prec(inputs_prec).detach()
        outputs_prev = self.networks_prev(inputs).detach()
        outputs, outputs_prec = self.networks(paddle.cat([inputs, inputs_prec], dim=0)).divide(2)
        loss, losses = self._compute_loss(outputs, outputs_prec, outdated_outputs_prec, pids, pids_prec, outputs_prev)
        loss.backward()
        self.optimizer.step()
        return losses

    def _compute_loss(self, outputs, outputs_prec, outdated_outputs_prec, pids, pids_prec, outputs_prev=None):
        pooled, preds = outputs["global"], outputs["preds"]
        pooled_prec, preds_prec = outputs_prec["global"], outputs_prec["preds"]
        pid_loss, triplet_loss = self.basic_criterion(paddle.cat([pooled, pooled_prec], dim=0),
                                                      paddle.cat([preds, preds_prec], dim=0),
                                                      paddle.cat([pids, pids_prec], dim=0)
                                                      )

        distillation_loss = self.distillation_criterion(outputs_prec['embedding'], outdated_outputs_prec['embedding'], pids_prec)
        distillation_loss = self.distillation_config["factor"] * distillation_loss

        domain_loss = self.domain_criterion(outputs['embedding'],
                                            outputs_prev['embedding'])
        domain_loss = self.co_distillation_config["factor"] * domain_loss

        loss = pid_loss + triplet_loss + distillation_loss + domain_loss
        return loss, [pid_loss.item(), triplet_loss.item(), distillation_loss.item(), domain_loss.item()]

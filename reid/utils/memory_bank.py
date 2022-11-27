# -*- coding: utf-8 -*-
# Time    : 2020/9/11 14:16
# Author  : Yichen Lu

import paddle


class MemoryBank(object):
    def __init__(self, max_size=1024):
        self.max_size = max_size
        self.features_queue = []
        self.labels_queue = []

    def push(self, features, labels=None):
        if len(self) + len(features) <= self.max_size:
            self.features_queue.append(features)
            self.labels_queue.append(labels)
        else:
            self.features_queue.pop(0)
            self.features_queue.append(features)

            self.labels_queue.pop(0)
            self.labels_queue.append(labels)

    def __len__(self):
        return sum([len(batch) for batch in self.features_queue]) if self.features_queue else 0

    def __call__(self, labels=False):
        if self.features_queue:
            if not labels:
                return paddle.concat(self.features_queue, axis=0)
            else:
                return paddle.concat(self.features_queue, axis=0), paddle.concat(self.labels_queue, axis=0)
        return None

